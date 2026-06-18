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

## When to use / not use

- **Use** for any warp-specialized kernel where roles are split across warps and coordinated by `mbarrier` / named-barrier / `__syncthreads` / pipeline objects: FlashAttention, GEMM with TMA producer + tensor-core consumer, MoE, fused epilogues, etc.
- **Language:** CuTe-DSL or CUDA C++ only. **Triton cannot do this** — it neither exposes warp specialization nor lets you read a timestamp and write it to shared/global memory at a chosen point.
- **Not** a replacement for Nsight Compute/Systems. Those give occupancy, achieved throughput, memory bandwidth, and instruction mix. This gives the *pipeline structure* (who waits for whom, and when) that ncu/nsys do not show per-warp-role.

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

## What to look for

- **Stall bars** (rendered gray/hatched): a warp blocked on a barrier. A consumer that stalls a lot is bottlenecked by its producer; a producer that never stalls is the bottleneck.
- **Dependency arrows** (producer `arrive` → consumer `wait`-exit): the arrow lands exactly where the stall ends → that producer was the binding constraint. If a consumer's wait-exit is *well after* the producer's signal, something else was binding.
- **Overlap**: do two role-rows have work spans at the *same x*? That concurrency is the win. Quantify it: `intersection(producerA_busy, producerB_busy) / busy_span`. Near-100% overlap between the two co-bottleneck resources means the pipeline is doing its job.
- **Early signalling**: a dependency arrow leaving the *middle* of a work bar means the warp publishes a partial result before finishing — a deliberate pipelining trick worth calling out.

## Constraints & caveats

- **CuTe-DSL / CUDA C++ only** (not Triton).
- **Per-SM clock → single CTA only.** Cross-SM timestamps are not comparable; restrict recording to one block (`blockIdx==0`) and shape the workload so only one CTA has work.
- **The stamps perturb timing** (each is a global store). Keep the event count small, put stamps off the critical path where possible, and trust **ratios/structure** over absolute cycle counts. Reproduce across a few runs (should be stable to <1%); cross-check cycles-per-iteration against a roofline estimate.
- Keep it gated (env var / null pointer) so production builds are byte-identical.
