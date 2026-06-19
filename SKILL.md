---
name: warp-specialization-report-skill
description: Profile a warp-specialized GPU kernel (CuTe-DSL or CUDA C++) by stamping the per-SM clock() register at each cross-warp synchronization point, then reconstruct a per-warp pipeline timeline (Gantt chart). Reveals where warps STALL, the cross-warp DATA DEPENDENCIES (which warp's output unblocks which), and how compute/memory phases OVERLAP. Use when the user wants to understand or visualize a warp-specialized kernel's runtime pipeline, find where warps wait on each other, or verify that producer/consumer warps actually run concurrently. Triggers include "profile the warp specialization", "pipeline timeline", "where does it stall", "how do the warps overlap", "visualize the producer/consumer pipeline". NOT applicable to Triton (cannot express warp specialization nor write timestamps to shared memory) and NOT a substitute for Nsight Compute/Systems for occupancy/throughput/memory-bandwidth analysis.
allowed-tools: "Bash Read Edit Write Grep Glob"
---

# Warp-Specialization Pipeline Profiling (clock() timeline)

**Goal:** turn a warp-specialized kernel's runtime into a per-warp Gantt timeline so you can *see* three things the usual profilers hide:

1. **Stalls** — when a warp is blocked waiting on a dependency (not doing work).
2. **Data dependencies** — which warp's output (a barrier `arrive`) unblocks which other warp's `wait`.
3. **Overlap** — whether the specialized warps (e.g. load / matmul / softmax / epilogue) actually run *concurrently*, which is the entire point of warp specialization.

The idea: every warp reads the **per-SM cycle counter** (`clock()` / `%clock`) at its key synchronization boundaries and writes the value to a global buffer. After the run you reconstruct the timeline offline.

**Use it as a closed loop, not just a picture.** The measurement (the 5 steps below) is the middle; bracket it with a **prediction** before and a **reconciliation** after:

> **Predict (feeds & speeds)** → **Measure (stamp → timeline)** → **Reconcile (predicted vs. measured)**

Predict which warp is the bottleneck and how much each one *should* idle from a back-of-envelope cycle budget; then check the measured timeline against that hypothesis. A timeline alone tells you *what* happened; the loop tells you *whether you understand why* — and every gap between predicted and measured is a finding. See **[Predict before you measure](#predict-before-you-measure-feeds--speeds)** and **[Reconcile](#reconcile-predicted-vs-measured)**.

## When to use / not use

- **Use** for any warp-specialized kernel where roles are split across warps and coordinated by `mbarrier` / named-barrier / `__syncthreads` / pipeline objects: FlashAttention, GEMM with TMA producer + tensor-core consumer, MoE, fused epilogues, etc.
- **Language:** CuTe-DSL or CUDA C++ only. **Triton cannot do this** — it neither exposes warp specialization nor lets you read a timestamp and write it to shared/global memory at a chosen point.
- **Not** a replacement for Nsight Compute/Systems. Those give occupancy, achieved throughput, memory bandwidth, and instruction mix. This gives the *pipeline structure* (who waits for whom, and when) that ncu/nsys do not show per-warp-role.

## Predict before you measure (feeds & speeds)

Before instrumenting anything, write a back-of-envelope **per-resource cycle budget** for one main-loop iteration. This is the "feeds and speeds" step (FlashAttention-4 §3.1.1): it costs minutes and turns the timeline from a pretty picture into a hypothesis test. Reason about **shared hardware resources**, not warps — a warp-specialized kernel is paced by a handful of them (tensor core / MMA, shared-memory bandwidth, the special-function "exp"/MUFU unit, TMA·DRAM bandwidth, CUDA-core FMA, …).

1. **List the resources** this kernel actually touches.
2. **Write a cyc/iter formula per resource** = `work ÷ throughput`. Work comes from the tile shape; throughput from the hardware (and the **precision** — low-bit speeds up MMA but *not* exp/softmax). E.g. an MMA: `FLOP ÷ (FLOP/cyc at that dtype)`; smem: `bytes ÷ (B/cyc)`; exp: `#exp-ops ÷ (ops/cyc)`.
3. **Plug in the tile shape.** The **largest** budget is the predicted lower bound on iteration time → that resource is the predicted **bottleneck**; the rest have slack.
4. **Map each resource → the warp role that drives it**, then predict each warp's idle:
   `predicted idle(warp) = 1 − T_resource(warp) / T_bottleneck`.
   That is your prediction of which timeline rows will be packed and which will show stall/slack — written down *before* you run.
5. **Note what the model omits** — reductions, format conversion / quantization, epilogue stores, barrier latency, pipeline fill. These are exactly where measured will exceed predicted (see Reconcile).

Output a one-line falsifiable hypothesis, e.g. *"resource R is the ceiling at ~T cyc/iter; warp Y idles ~Z%."* A worked multi-resource example (FA-4's MMA/smem/exp roofline, and how low precision shifts the bottleneck onto softmax) is in [`README.md`](README.md).

## Method (5 steps)

### 1. Identify the warp roles
Find how the kernel maps `warp_id` → role (producer/consumer/etc.) and list every cross-warp sync point in each role's loop (`mbarrier.arrive` / `mbarrier.try_wait` / pipeline `producer_acquire` / `consumer_wait` / `__syncthreads`). These sync points are the timeline's edges.

### 2. Stamp clock() at each sync boundary — both sides of every blocking wait
- After a **producer signal** (`arrive` / `commit`) → marks "output ready at time T".
- **Before AND after a blocking wait** (`wait` / `acquire`) → the gap `exit − enter` *is the stall*. This is the key to seeing stalls; a single stamp can't show them.
- Around the **work** itself (issue of a GEMM, start/end of a compute loop) → marks active spans.

### 3. Write stamps to a small global buffer, single-CTA
- Flat `int32` buffer indexed `slot = (role*N_EVT + evt)*MAX_IT + it`, where `it` is the loop iteration. `clock()` is 32-bit and wraps, but a single tile's span is far under 2³² cycles, so use `clock()` (cheaper) or `clock64()`.
- **`clock()` is per-SM and unsynchronized across SMs.** So run a workload that lands on **one CTA / one SM** (one tile, one head, one batch) — then all warps share one clock domain and timestamps are directly comparable. (Disable persistent scheduling / multi-CTA cooperation for the profiling build.)
- Gate everything behind a flag so it compiles out in normal builds. One writer per warp (`elect_one` / `lane==0`); redundant writers in a warpgroup hit the same slot harmlessly.

CuTe-DSL pattern:
```python
N_EVT, MAX_IT = 16, 64   # per role; size the buffer = N_ROLE*N_EVT*MAX_IT int32
@cute.jit
def prof_stamp(mProf, role, evt: cutlass.Constexpr, it):
    if cutlass.const_expr(mProf is not None):        # compiles out when off
        if it < MAX_IT:
            with cute.arch.elect_one():              # one writer per warp
                mProf[(role * N_EVT + evt) * MAX_IT + it] = cute.arch.clock()
```
CUDA C++ pattern:
```cuda
__device__ __forceinline__
void prof_stamp(int* buf, int role, int evt, int it) {        // buf=nullptr when off
    if (buf && blockIdx.x == 0 && (threadIdx.x & 31) == 0 && it < MAX_IT)
        buf[(role*N_EVT + evt)*MAX_IT + it] = (int)clock();   // %clock, per-SM
}
```
Place `prof_stamp(...)` immediately before/after each `wait`/`arrive` and around each work region. Thread the buffer pointer in as an extra (optional) kernel argument.

### 4. Run and dump
Launch once on the single-CTA workload, copy the buffer to host, reshape to `(N_ROLE, N_EVT, MAX_IT)`, reinterpret as `uint32`, subtract the global min to get cycles-since-start. Save as `.npz`.

### 5. Plot the timeline
Use [`helpers/plot_timeline.py`](helpers/plot_timeline.py) with a small JSON spec mapping `(role, evt)` slots to **active spans**, **stall spans** (begin/exit of each wait), **markers**, and **dependency arrows** (producer `(role,evt)` → consumer `(role,evt)`). It renders the Gantt + arrows and prints per-warp stall % and a compute-overlap ratio. See [`README.md`](README.md) for a complete worked example.

## Reading the chart: three states per row (do not conflate them)

Every row has **three** visual states, and the difference is *measured vs. not-measured* — not "more stall vs. less stall":

1. **Colored bar = instrumented active work** — bracketed by a stamp pair (e.g. a span's `beg`/`end`).
2. **Gray hatched = a *measured* stall** — the warp was provably parked on a barrier: the kernel stamped `clock()` immediately *before and after* a specific `wait`, and the bar width is that exact enter→exit delta. This is the only state that proves blocking.
3. **Light (untracked) underlay = an *un-instrumented* interval** — no stamp pair brackets it, so the tool draws nothing definite. It is **not** a stall and must not be read as one. It is usually un-instrumented productive work (an epilogue between two stamped events, or an async copy where only the *issue instant* was stamped, not its duration), or genuine slack. Plain white = the role had no stamps there at all.

> The plotter lays a faint underlay across each row's stamped span precisely so blanks become a labeled category instead of invisible whitespace. When you see a row that is mostly light, that means *we did not instrument it*, **not** that the warp was idle/blocked — go add stamps there if it matters.

## What to look for

- **Stall bars** (gray/hatched — a *measured* wait): a warp blocked on a barrier. A consumer that stalls a lot is bottlenecked by its producer; a producer that never stalls is the bottleneck. (Light underlay is *not* this — see above.)
- **Dependency arrows** (producer `arrive` → consumer `wait`-exit): the arrow lands exactly where the stall ends → that producer was the binding constraint. If a consumer's wait-exit is *well after* the producer's signal, something else was binding.
- **Overlap**: do two role-rows have work spans at the *same x*? That concurrency is the win. Quantify it: `intersection(producerA_busy, producerB_busy) / busy_span`. Near-100% overlap between the two co-bottleneck resources means the pipeline is doing its job.
- **Early signalling**: a dependency arrow leaving the *middle* of a work bar means the warp publishes a partial result before finishing — a deliberate pipelining trick worth calling out.

## Reconcile: predicted vs. measured

Close the loop: put the [prediction](#predict-before-you-measure-feeds--speeds) next to the timeline numbers and explain the difference. This is where the actual understanding lands.

1. **Build a small table** — per warp/resource: predicted busy cyc, predicted idle %, **measured** stall % (printed by the plotter), measured cyc/iter.
2. **Check direction & ranking first, magnitude second.** "Does the predicted bottleneck actually have the least idle? Does idle move the predicted way when I change precision/tile?" Ranking is robust even when absolute cycles are perturbed by the stamps.
3. **Account for the residual** — measured is almost always *worse* than the ideal roofline. Usual suspects:
   - **Instrumentation overhead** — the stamps are global stores on the critical path; they inflate absolute stall. Keep instrumentation *identical* across configs so cross-config deltas stay valid, and trust **ratios** over absolutes.
   - **Pipeline fill / drain & prologue** — first/last iterations aren't steady state; the ideal budget assumes steady state.
   - **Untracked work the roofline ignored** — the **light (untracked) rows**. If a row is mostly light *and* measured cyc/iter exceeds the per-resource budget, that gap is real work the feeds&speeds model didn't count (e.g. softmax's row-max reduction, scale-subtract, P convert/requant, write-back) → either add the term to the model or instrument it, then re-reconcile.
   - **Dependency / barrier latency** — a consumer's wait-exit lands some slack after the producer's signal even when nothing is "the bottleneck."
4. **A mismatch has two possible causes — and telling them apart is the whole skill.** When measured ≠ predicted, do **not** assume the kernel is at fault, and do **not** assume your analysis is. Either is possible:
   - **(a) The analysis is wrong or incomplete** — you missed a resource, mis-estimated a throughput, ignored a serialization/dependency, or omitted real work (the *untracked* rows are the usual tell). Fix: add the missing term/edge to the feeds&speeds model and re-reconcile. The model should explain the kernel, not the other way around.
   - **(b) The kernel implementation is leaving performance on the table** — the analysis is right, but the kernel under-overlaps, stalls on avoidable dependencies, picks a bad tile, or serializes warps that *could* run concurrently. Fix: that gap is an optimization opportunity (the design *should* hit the predicted bound but doesn't).
   **Default to suspecting (a) first** — re-derive the budget, account for every untracked region, confirm the bottleneck resource — because a confident "the kernel is bad" built on a wrong analysis is the worst outcome. Only once the model is airtight and still under-performs do you have evidence for (b). Either way the mismatch — not the agreement — is the most valuable output; chase it until one of (a)/(b) explains it.

A worked predicted-vs-measured table (FA-4 at BF16 / FP8 / NVFP4, where low precision shrinks `T_MMA` but not `T_exp` and pushes the kernel softmax-bound) is in [`README.md`](README.md).

> **Iterating this skill:** the long-term goal is for the agent to get *good at the feeds&speeds analysis itself* — choosing the right resources, formulas, and warp→resource mapping for an unfamiliar kernel. Treat every reconciliation as training signal: each time a mismatch turns out to be cause (a), note what the analysis missed so the next prediction includes it. The accuracy of the prediction is the thing being improved, because only a trustworthy prediction lets you make the (a)-vs-(b) call.

## Constraints & caveats

- **CuTe-DSL / CUDA C++ only** (not Triton).
- **Per-SM clock → single CTA only.** Cross-SM timestamps are not comparable; restrict recording to one block (`blockIdx==0`) and shape the workload so only one CTA has work.
- **The stamps perturb timing** (each is a global store). Keep the event count small, put stamps off the critical path where possible, and trust **ratios/structure** over absolute cycle counts. Reproduce across a few runs (should be stable to <1%); cross-check cycles-per-iteration against a roofline estimate.
- Keep it gated (env var / null pointer) so production builds are byte-identical.
