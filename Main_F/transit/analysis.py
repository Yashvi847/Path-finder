"""Empirical verification of heuristic properties (admissibility, consistency)."""
from __future__ import annotations
from typing import Dict
from .search import successors, dijkstra
from .heuristics import make_heuristic


def consistency_report(net, dest, cm, h) -> Dict[str, float]:
    """
    Check  h(u) <= w(u,v) + h(v)  on EVERY edge of the state graph.
    If it holds and h(goal)=0 then h is consistent and therefore admissible
    for this goal (a proof, not a sample).
    """
    violations, worst, n_edges = 0, 0.0, 0
    states = [(s, None) for s in net.stops]
    for s in net.stops:
        for rid in net.routes_at[s]:
            states.append((s, rid))
    for st in states:
        for ns, _, t, f, tr in successors(net, st):
            n_edges += 1
            w = cm.weight(t, tr, f)
            gap = h(st[0]) - (w + h(ns[0]))
            if gap > 1e-9:
                violations += 1
                worst = max(worst, gap)
    return dict(edges_checked=n_edges, violations=violations, worst_violation=worst,
                h_goal=h(dest))


def admissibility_report(net, dest, cm, h, origins) -> Dict[str, float]:
    """Compare h(origin) against the true optimal cost for sampled origins."""
    viol, worst, ratios = 0, 0.0, []
    for o in origins:
        if o == dest:
            continue
        opt = dijkstra(net, o, dest, cm).objective
        if h(o) > opt + 1e-9:
            viol += 1
            worst = max(worst, h(o) - opt)
        if opt > 0:
            ratios.append(h(o) / opt)
    return dict(sampled=len(ratios), violations=viol, worst_overestimate=worst,
                mean_h_over_opt=sum(ratios) / max(1, len(ratios)))
