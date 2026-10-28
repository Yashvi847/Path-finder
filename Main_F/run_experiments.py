"""
Experimental evaluation (Section 8).  Run:  python run_experiments.py
Writes CSVs to results/ and figures to results/figures/.
"""
import random, math, os, sys
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from transit import *
from transit.analysis import consistency_report

OUT = "results"; FIG = f"{OUT}/figures"
os.makedirs(FIG, exist_ok=True)
SEED = 42
N_PAIRS = 30            # 10 short / 10 medium / 10 long per network
REPEATS = 3
ALGS = list(ALGORITHMS)
EXACT = {"BFS", "Dijkstra", "A* (admissible)", "A* (0.5x, weak)", "A* (Manhattan)", "A* (2x, inadmissible)"}


def sample_pairs(net, k, seed):
    rng = random.Random(seed)
    cand = []
    for _ in range(600):
        o, d = rng.sample(range(net.n_stops), 2)
        cand.append((net.dist(o, d), o, d))
    cand.sort()
    third = len(cand) // 3
    out = []
    for cls, chunk in zip(("short", "medium", "long"),
                          (cand[:third], cand[third:2 * third], cand[2 * third:])):
        for dd, o, d in rng.sample(chunk, k // 3):
            out.append((o, d, dd, cls))
    return out


def benchmark(cm_name, cm):
    rows = []
    for size in ("small", "medium", "large"):
        net = make_benchmark(size, SEED)
        pairs = sample_pairs(net, N_PAIRS, SEED)
        print(f"[{cm_name}] {size}: {net.n_stops} stops, {net.n_states} states", flush=True)
        for i, (o, d, dd, cls) in enumerate(pairs):
            ref = None
            for lab in ALGS:
                r = measure(net, lab, o, d, cm, REPEATS, memory=True)
                if lab == "Dijkstra":
                    ref = r.objective
                rows.append(dict(cost_model=cm_name, size=size, stops=net.n_stops,
                                 states=net.n_states, pair=i, origin=o, dest=d,
                                 euclid_km=dd, dist_class=cls, algorithm=lab,
                                 found=r.found, generated=r.generated,
                                 expanded=r.expanded, max_frontier=r.max_frontier,
                                 runtime_ms=r.runtime_ms, peak_mem_kb=r.peak_mem_kb,
                                 travel_time=r.total_time, transfers=r.transfers,
                                 fare=r.total_fare, objective=r.objective, ref=None))
            # fill optimal reference for this pair
            for row in rows[-len(ALGS):]:
                row["ref"] = ref
    df = pd.DataFrame(rows)
    df["optimal"] = (df.objective - df.ref).abs() < 1e-6
    df["gap_pct"] = 100 * (df.objective - df.ref) / df.ref.where(df.ref > 0)
    return df


def summarise(df):
    g = df.groupby(["cost_model", "size", "algorithm"], sort=False)
    s = g.agg(stops=("stops", "first"), generated=("generated", "mean"),
              expanded=("expanded", "mean"), runtime_ms=("runtime_ms", "mean"),
              peak_mem_kb=("peak_mem_kb", "mean"), travel_time=("travel_time", "mean"),
              transfers=("transfers", "mean"), fare=("fare", "mean"),
              optimal_pct=("optimal", lambda x: 100 * x.mean()),
              mean_gap_pct=("gap_pct", "mean"), max_gap_pct=("gap_pct", "max")).reset_index()
    return s


def plots(df, tag):
    d = df[df.cost_model == tag]
    order = ["small", "medium", "large"]
    # 1. expanded & runtime per algorithm & size
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
    for a, (col, title) in zip(ax, [("expanded", "Nodes expanded"), ("runtime_ms", "Runtime (ms)"),
                                     ("peak_mem_kb", "Peak memory (KiB)")]):
        p = d.pivot_table(index="size", columns="algorithm", values=col).loc[order][ALGS]
        p.plot(kind="bar", ax=a, logy=True, rot=0, legend=False)
        a.set_title(f"{title}  [{tag}]"); a.set_xlabel("")
    ax[0].legend(fontsize=7)
    plt.tight_layout(); plt.savefig(f"{FIG}/{tag}_efficiency.png", dpi=140); plt.close()

    # 2. scalability
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    for lab in ["BFS", "Dijkstra", "A* (admissible)"]:
        s = d[d.algorithm == lab].groupby("stops")[["expanded", "runtime_ms"]].mean()
        ax[0].plot(s.index, s.expanded, "o-", label=lab)
        ax[1].plot(s.index, s.runtime_ms, "o-", label=lab)
    for a, t in zip(ax, ["Nodes expanded", "Runtime (ms)"]):
        a.set_xscale("log"); a.set_yscale("log"); a.set_xlabel("stops |V|"); a.set_title(t + f" vs size [{tag}]")
        a.grid(alpha=.3)
    ax[0].legend(); plt.tight_layout(); plt.savefig(f"{FIG}/{tag}_scalability.png", dpi=140); plt.close()

    # 3. A* saving vs OD distance (large)
    L = d[d["size"] == "large"]
    dj = L[L.algorithm == "Dijkstra"].set_index("pair")
    as_ = L[L.algorithm == "A* (admissible)"].set_index("pair")
    red = 100 * (1 - as_.expanded / dj.expanded)
    plt.figure(figsize=(6, 4.2))
    plt.scatter(dj.euclid_km, red, c="tab:green")
    plt.xlabel("Origin-destination straight-line distance (km)")
    plt.ylabel("Nodes expanded saved by A* (%)")
    plt.title(f"A* vs Dijkstra, large network [{tag}]"); plt.grid(alpha=.3)
    plt.tight_layout(); plt.savefig(f"{FIG}/{tag}_astar_saving.png", dpi=140); plt.close()

    # 4. optimality gap
    p = d.groupby("algorithm")["gap_pct"].mean().reindex(ALGS)
    plt.figure(figsize=(7, 3.8)); p.plot(kind="bar", rot=20, color="tab:red")
    plt.ylabel("Mean objective gap vs optimal (%)"); plt.title(f"Route quality [{tag}]")
    plt.tight_layout(); plt.savefig(f"{FIG}/{tag}_optimality.png", dpi=140); plt.close()


def heuristic_study(cm_name, cm):
    """Formal consistency check + empirical effect of heuristic variants (medium net)."""
    net = make_benchmark("medium", SEED)
    pairs = sample_pairs(net, N_PAIRS, SEED)
    rows = []
    for lab in ["Dijkstra", "A* (0.5x, weak)", "A* (admissible)", "A* (Manhattan)", "A* (2x, inadmissible)"]:
        spec = ALGORITHMS[lab]
        kind = spec.get("kind", "zero"); w = spec.get("weight", 1.0)
        viol = 0; worst = 0.0; checked = 0
        for (o, d, _, _) in pairs[:10]:
            h = make_heuristic(net, d, cm, kind, w)
            rep = consistency_report(net, d, cm, h)
            viol += rep["violations"]; worst = max(worst, rep["worst_violation"]); checked += rep["edges_checked"]
        rows.append(dict(cost_model=cm_name, heuristic=lab, kind=kind, weight=w,
                         edges_checked=checked, consistency_violations=viol,
                         worst_violation=round(worst, 4)))
    return pd.DataFrame(rows)


def tradeoff_study():
    """How the optimisation criterion changes the recommended journey (A*, medium)."""
    net = make_benchmark("medium", SEED)
    pairs = sample_pairs(net, N_PAIRS, SEED)
    rows = []
    for name, cm in PRESETS.items():
        for (o, d, dd, cls) in pairs:
            r = run(net, "A* (admissible)", o, d, cm)
            rows.append(dict(criterion=name, travel_time=r.total_time, fare=r.total_fare,
                             transfers=r.transfers, expanded=r.expanded))
    return pd.DataFrame(rows).groupby("criterion", sort=False).mean().round(2).reset_index()


if __name__ == "__main__":
    frames, hs = [], []
    for name in ("Fastest", "Balanced"):
        frames.append(benchmark(name, PRESETS[name]))
        hs.append(heuristic_study(name, PRESETS[name]))
    df = pd.concat(frames, ignore_index=True)
    df.to_csv(f"{OUT}/raw_results.csv", index=False)
    summ = summarise(df); summ.round(3).to_csv(f"{OUT}/summary.csv", index=False)
    pd.concat(hs).to_csv(f"{OUT}/heuristic_study.csv", index=False)
    tradeoff_study().to_csv(f"{OUT}/criterion_tradeoff.csv", index=False)
    for t in ("Fastest", "Balanced"):
        plots(df, t)
    pd.set_option("display.width", 220); pd.set_option("display.max_columns", 30)
    print(summ.round(2).to_string())
    print(pd.concat(hs).to_string())
    print(tradeoff_study().to_string())
