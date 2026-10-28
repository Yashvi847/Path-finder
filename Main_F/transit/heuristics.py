"""
Heuristics for the route-aware transit search.

Straight-line (Euclidean) heuristic for  C = alpha*T + beta*R + gamma*K :

    h(n) = alpha * 60 * d(n, goal) / v_max   +   gamma * c_min * d(n, goal)

* d(n, goal)  straight-line distance in km
* v_max       fastest speed in the network (km/h)  -> time lower bound
* c_min       cheapest fare per km in the network  -> fare lower bound
* beta term   0 (transfers cannot be lower-bounded from geometry alone)

Admissibility
    For any path P from n to the goal, sum of edge euclid lengths >= d(n,goal)
    (triangle inequality).  Each edge satisfies time >= euclid/v_max and
    fare >= c_min*euclid, so  T(P) >= 60 d / v_max  and  K(P) >= c_min d.
    Waiting, boarding fees and transfer penalties are >= 0, so
    C(P) >= h(n).  Hence h never overestimates.

Consistency
    For every ride edge (u,v):  h(u) - h(v) <= (alpha*60/v_max + gamma*c_min)*d(u,v)
    <= alpha*time(u,v) + gamma*fare(u,v) = w(u,v).  Transfer / boarding steps
    do not change the stop, so h is unchanged and the inequality holds as
    w >= 0.  Hence h is consistent and A* never needs to re-open nodes.
"""
from __future__ import annotations

from typing import Callable, Dict


def make_heuristic(net, dest: int, cm, kind: str = "euclid",
                   weight: float = 1.0) -> Callable[[int], float]:
    """
    kind   : "euclid"    - admissible + consistent when weight <= 1
             "manhattan" - L1 distance; may overestimate (inadmissible)
             "zero"      - h = 0  (A* degenerates to Dijkstra)
    weight : multiplier on h (w > 1 -> weighted A*, bounded-suboptimal;
             w < 1 -> still admissible but less informed)
    """
    if kind == "zero":
        return lambda s: 0.0
    vmax, cmin = net.bounds()
    per_km = cm.alpha * 60.0 / vmax + cm.gamma * cmin
    dist = net.dist if kind == "euclid" else net.manhattan
    table: Dict[int, float] = {s: weight * per_km * dist(s, dest) for s in net.stops}
    return table.__getitem__
