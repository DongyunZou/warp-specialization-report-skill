"""Generic per-warp pipeline timeline from clock() stamps.

Reads a stamp buffer and a JSON spec, renders a Gantt chart (active spans, stall
spans, markers) plus data-dependency arrows, and prints per-warp stall % and a
compute-overlap ratio. Kernel-agnostic: all kernel specifics live in the spec.

Usage:
    python plot_timeline.py --trace run.npz --spec spec.json --out timeline.png

--trace : .npz containing array `prof` of shape (n_role, n_evt, max_it), int32.
          Values are uint32 clock() readings; 0 means "empty slot".
          (Optionally a `meta` JSON string; printed if present.)

Spec JSON (every evt id indexes the evt axis; spans pair beg/end at the same it):
{
  "roles":   ["Load", "Softmax0", "MMA", "Softmax1", "Correction"],   # top->bottom
  "active":  [{"role":"Softmax0","beg":1,"end":2,"color":"#1b7837","label":"exp"}],
  "stall":   [{"role":"Softmax0","beg":0,"end":1}],
  "markers": [{"role":"Load","evt":0,"color":"#d62728","marker":"^"}],
  "deps":    [{"prod":"MMA","pevt":0,"cons":"Softmax0","cevt":1,
               "color":"#3b6fb6","label":"S->softmax"}],
  "overlap": [["MMA",0,1],["Softmax0",1,2]]   # optional: two busy-span sets to intersect
}

Tip: order `roles` so every dependency connects ADJACENT rows (put the central
producer/consumer hub in the middle) — then arrows stay in the clean gaps.
"""
from __future__ import annotations
import argparse
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch, FancyArrowPatch
from matplotlib import patheffects as pe

H = 0.54
C_STALL, C_STALL_EC = "#a6a6a6", "#5a5a5a"


def ev(raw, role, e, t0):
    row = raw[role, e]
    return {int(i): int(row[i]) - t0 for i in np.nonzero(row)[0]}


def merge(ivals):
    if not ivals:
        return []
    ivals = sorted(ivals)
    out = [list(ivals[0])]
    for s, e in ivals[1:]:
        if s > out[-1][1]:
            out.append([s, e])
        else:
            out[-1][1] = max(out[-1][1], e)
    return [(s, e) for s, e in out]


def releaser(prod, t):
    cand = [p for p in prod if p <= t + 60]
    return max(cand) if cand else (min(prod, key=lambda p: abs(p - t)) if prod else None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace", required=True)
    ap.add_argument("--spec", required=True)
    ap.add_argument("--out", default="timeline.png")
    ap.add_argument("--zoom", type=float, default=0.0,
                    help="if >0, second panel zooms to this fraction of the span around the middle")
    args = ap.parse_args()

    z = np.load(args.trace, allow_pickle=True)
    raw = z["prof"].astype(np.uint64)
    t0 = int(raw[raw != 0].min())
    total = int(raw[raw != 0].max()) - t0
    spec = json.loads(open(args.spec).read())
    roles = spec["roles"]                                   # display order, top->bottom
    yof = {r: len(roles) - 1 - i for i, r in enumerate(roles)}
    rid = spec.get("role_id") or {r: i for i, r in enumerate(roles)}  # name -> prof role index

    def bar(ax, y, segs, fc, tlo, thi, hatch=None, ec="white", lw=0.4, z=2, alpha=1.0):
        if tlo is not None:
            segs = [(s, w) for s, w in segs if s + w >= tlo and s <= thi]
        if segs:
            ax.broken_barh(segs, (y - H / 2, H), facecolors=fc, edgecolors=ec,
                           linewidth=lw, hatch=hatch, zorder=z, alpha=alpha)

    def arrow(ax, x0, yp, x1, yc, color):
        e0 = yp + (H / 2 if yc > yp else -H / 2)
        e1 = yc + (-H / 2 if yc > yp else H / 2)
        rad = 0.16 if yc > yp else -0.16
        p = FancyArrowPatch((x0, e0), (x1, e1), connectionstyle=f"arc3,rad={rad}",
                            arrowstyle="-|>", mutation_scale=10, color=color, lw=1.3,
                            zorder=11, alpha=0.95, shrinkA=1, shrinkB=2)
        p.set_path_effects([pe.withStroke(linewidth=3.0, foreground="white")])
        ax.add_patch(p)

    def draw(ax, tlo, thi, title, arrows):
        for a in spec.get("active", []):
            y, beg, end = yof[a["role"]], ev(raw, rid[a["role"]], a["beg"], t0), ev(raw, rid[a["role"]], a["end"], t0)
            for it in beg:
                if it in end and end[it] > beg[it]:
                    bar(ax, y, [(beg[it], end[it] - beg[it])], a["color"], tlo, thi, lw=0.4, z=2)
        for st in spec.get("stall", []):
            y, beg, end = yof[st["role"]], ev(raw, rid[st["role"]], st["beg"], t0), ev(raw, rid[st["role"]], st["end"], t0)
            for it in beg:
                if it in end and end[it] > beg[it]:
                    bar(ax, y, [(beg[it], end[it] - beg[it])], C_STALL, tlo, thi,
                        hatch="////", ec=C_STALL_EC, lw=0.4, z=4, alpha=0.9)
        for m in spec.get("markers", []):
            y = yof[m["role"]]
            xs = [t for t in ev(raw, rid[m["role"]], m["evt"], t0).values() if tlo is None or tlo <= t <= thi]
            ax.scatter(xs, [y] * len(xs), marker=m.get("marker", "o"), s=26,
                       c=m["color"], zorder=6, edgecolors="black", linewidths=0.3)
        if arrows:
            for d in spec.get("deps", []):
                prod = sorted(ev(raw, rid[d["prod"]], d["pevt"], t0).values())
                for it, t in ev(raw, rid[d["cons"]], d["cevt"], t0).items():
                    if (tlo is None or tlo <= t <= thi):
                        s = releaser(prod, t)
                        if s is not None:
                            arrow(ax, s, yof[d["prod"]], t, yof[d["cons"]], d["color"])
        ax.set_yticks(list(yof.values()))
        ax.set_yticklabels(list(yof.keys()), fontsize=9)
        ax.set_ylim(-0.6, len(roles) - 0.4)
        if tlo is not None:
            ax.set_xlim(tlo, thi)
        ax.set_xlabel("SM clock cycles (relative to first stamp)", fontsize=9)
        ax.set_title(title, fontsize=10)
        ax.grid(axis="x", linestyle=":", alpha=0.4)

    # stall % per role (union of its stall intervals)
    print(f"total span: {total} cyc")
    for st_role in roles:
        ivs = []
        for st in spec.get("stall", []):
            if st["role"] != st_role:
                continue
            beg, end = ev(raw, rid[st_role], st["beg"], t0), ev(raw, rid[st_role], st["end"], t0)
            ivs += [(beg[it], end[it]) for it in beg if it in end and end[it] > beg[it]]
        s = sum(e - b for b, e in merge(ivs))
        if ivs:
            print(f"  {st_role:14s} stalled {s:7d} cyc ({100*s/max(total,1):.1f}%)")
    if "overlap" in spec and len(spec["overlap"]) == 2:
        sets = []
        for role, b, e in spec["overlap"]:
            beg, end = ev(raw, rid[role], b, t0), ev(raw, rid[role], e, t0)
            sets.append(merge([(beg[it], end[it]) for it in beg if it in end and end[it] > beg[it]]))
        a, b = sets
        i = j = ov = 0
        while i < len(a) and j < len(b):
            lo, hi = max(a[i][0], b[j][0]), min(a[i][1], b[j][1])
            if hi > lo:
                ov += hi - lo
            if a[i][1] < b[j][1]:
                i += 1
            else:
                j += 1
        busy = sum(e - s for s, e in a)
        print(f"  overlap {spec['overlap'][0][0]} <-> {spec['overlap'][1][0]}: "
              f"{ov} cyc = {100*ov/max(busy,1):.1f}% of first busy span")

    npanels = 2 if args.zoom > 0 else 1
    fig, axes = plt.subplots(npanels, 1, figsize=(15, 4.2 * npanels), squeeze=False)
    draw(axes[0][0], None, None, "Warp-specialized pipeline timeline — full (gray-hatched = STALL)", arrows=(npanels == 1))
    if npanels == 2:
        w = total * args.zoom
        tlo = max(0, total / 2 - w / 2)
        draw(axes[1][0], tlo, tlo + w, "Zoom — arrows = data dependencies (producer signal -> consumer wait-exit)", arrows=True)

    legend = ([Patch(fc=a["color"], label=f"{a['role']}: {a.get('label','active')}") for a in spec.get("active", [])]
              + [Patch(fc=C_STALL, ec=C_STALL_EC, hatch="////", label="STALL")]
              + [plt.Line2D([], [], color=d["color"], lw=1.6, label="dep: " + d.get("label", f"{d['prod']}->{d['cons']}"))
                 for d in spec.get("deps", [])])
    fig.legend(handles=legend, loc="lower center", ncol=5, fontsize=8, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(args.out, dpi=150, bbox_inches="tight")
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
