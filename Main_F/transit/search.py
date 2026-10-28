"""
Search algorithms over the route-aware transit graph.

State          s = (stop, route)   (route is None before boarding)
Actions        board a route at the origin | ride to next stop | transfer
Goal test      s.stop == destination
Step weight    w = alpha*time + beta*transfers + gamma*fare      (C = aT + bR + gK)
"""
from __future__ import annotations

import heapq
import time as _time
import tracemalloc
from collections import deque
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

from .network import TransitNetwork
from .heuristics import make_heuristic

State = Tuple[int, Optional[str]]


# --------------------------------------------------------------------------
# Cost model
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class CostModel:
    """C = alpha*T + beta*R + gamma*K   (time [min], transfers, fare)."""
    alpha: float = 1.0
    beta: float = 0.0
    gamma: float = 0.0
    name: str = "custom"

    def weight(self, t: float, transfers: int, fare: float) -> float:
        return self.alpha * t + self.beta * transfers + self.gamma * fare


PRESETS: Dict[str, CostModel] = {
    "Fastest": CostModel(1.0, 0.0, 0.0, "Fastest"),
    "Cheapest": CostModel(0.0, 0.0, 1.0, "Cheapest"),
    "Fewest transfers": CostModel(0.05, 10.0, 0.01, "Fewest transfers"),
    "Balanced": CostModel(1.0, 5.0, 0.5, "Balanced"),
}


# --------------------------------------------------------------------------
# Result container
# --------------------------------------------------------------------------

@dataclass
class SearchResult:
    algorithm: str
    found: bool = False
    steps: list = field(default_factory=list)   # (action, from, to, t, fare, transfers)
    stops: List[int] = field(default_factory=list)
    total_time: float = float("inf")
    total_fare: float = float("inf")
    transfers: int = 0
    objective: float = float("inf")
    generated: int = 0
    expanded: int = 0
    max_frontier: int = 0
    runtime_ms: float = 0.0
    peak_mem_kb: float = 0.0
    explored: List[int] = field(default_factory=list)   # stops expanded

    def legs(self, net: TransitNetwork) -> List[dict]:
        """Human-readable itinerary grouped by route."""
        legs: List[dict] = []
        for action, a, b, t, f, tr in self.steps:
            if action in ("board", "transfer"):
                rid = b[1]
                legs.append(dict(route=rid, mode=net.routes[rid].mode,
                                 from_stop=a[0], to_stop=a[0], hops=0,
                                 wait=t, ride=0.0, fare=f,
                                 is_transfer=(action == "transfer")))
            else:
                leg = legs[-1]
                leg["to_stop"] = b[0]
                leg["hops"] += 1
                leg["ride"] += t
                leg["fare"] += f
        for leg in legs:
            leg["minutes"] = leg["wait"] + leg["ride"]
        return legs


# --------------------------------------------------------------------------
# Successor function
# --------------------------------------------------------------------------

def successors(net: TransitNetwork, state: State):
    """Yield (next_state, action, minutes, fare, transfers)."""
    v, r = state
    if r is None:
        for rid in net.routes_at[v]:
            rt = net.routes[rid]
            yield (v, rid), "board", rt.wait_min, rt.boarding_fee, 0
        return
    for (w, t, f) in net.rides.get((v, r), ()):
        yield (w, r), "ride", t, f, 0
    for rid in net.routes_at[v]:
        if rid != r:
            rt = net.routes[rid]
            yield (v, rid), "transfer", rt.wait_min, rt.boarding_fee, 1


def _finish(res: SearchResult, steps, cm: CostModel) -> SearchResult:
    res.found = True
    res.steps = steps
    res.total_time = sum(s[3] for s in steps)
    res.total_fare = sum(s[4] for s in steps)
    res.transfers = sum(s[5] for s in steps)
    res.objective = cm.weight(res.total_time, res.transfers, res.total_fare)
    stops = [steps[0][1][0]] if steps else []
    for _, _, b, *_ in steps:
        if b[0] != stops[-1]:
            stops.append(b[0])
    res.stops = stops
    return res


# --------------------------------------------------------------------------
# A* / Dijkstra  (Dijkstra == A* with h = 0)
# --------------------------------------------------------------------------

def a_star(net: TransitNetwork, origin: int, dest: int, cm: CostModel,
           h: Optional[Callable[[int], float]] = None,
           name: str = "A*") -> SearchResult:
    """
    Best-first search with f(n) = g(n) + h(n) on the route-aware state graph.
    With a consistent h the first goal popped is optimal; with an
    inconsistent / inadmissible h the algorithm still terminates but may
    return a sub-optimal path (lazy re-opening of states is supported).
    """
    res = SearchResult(name)
    t0 = _time.perf_counter()
    h = h or (lambda s: 0.0)

    start: State = (origin, None)
    g: Dict[State, float] = {start: 0.0}
    parent: Dict[State, Optional[tuple]] = {start: None}
    counter = 0
    pq = [(h(origin), 0.0, counter, start)]
    res.generated = 1
    explored = set()

    while pq:
        res.max_frontier = max(res.max_frontier, len(pq))
        f, neg_g, _, s = heapq.heappop(pq)
        gs = -neg_g
        if gs > g[s] + 1e-12:
            continue                      # stale queue entry
        res.expanded += 1
        explored.add(s[0])
        if s[0] == dest:
            steps = []
            cur = s
            while parent[cur] is not None:
                p, action, t, fare, tr = parent[cur]
                steps.append((action, p, cur, t, fare, tr))
                cur = p
            steps.reverse()
            res.explored = list(explored)
            res.runtime_ms = (_time.perf_counter() - t0) * 1000
            return _finish(res, steps, cm)
        for ns, action, t, fare, tr in successors(net, s):
            ng = gs + cm.weight(t, tr, fare)
            if ng < g.get(ns, float("inf")) - 1e-12:
                g[ns] = ng
                parent[ns] = (s, action, t, fare, tr)
                counter += 1
                heapq.heappush(pq, (ng + h(ns[0]), -ng, counter, ns))
                res.generated += 1

    res.explored = list(explored)
    res.runtime_ms = (_time.perf_counter() - t0) * 1000
    return res


def dijkstra(net, origin, dest, cm):
    return a_star(net, origin, dest, cm, None, name="Dijkstra")


# --------------------------------------------------------------------------
# BFS baseline (uninformed, ignores weights)
# --------------------------------------------------------------------------

def bfs(net: TransitNetwork, origin: int, dest: int, cm: CostModel) -> SearchResult:
    """
    Breadth-first search on the *stop* graph: finds the path with the fewest
    hops, ignoring time / fare.  The stop path is then "priced" with the best
    possible assignment of routes to hops (small DP), so that BFS can be
    compared with the weighted algorithms using the same metrics.
    """
    res = SearchResult("BFS")
    t0 = _time.perf_counter()
    parent = {origin: None}
    q = deque([origin])
    res.generated = 1
    explored = []
    found = origin == dest
    while q and not found:
        res.max_frontier = max(res.max_frontier, len(q))
        u = q.popleft()
        res.expanded += 1
        explored.append(u)
        for v in sorted(net.adjacency[u]):
            if v not in parent:
                parent[v] = u
                res.generated += 1
                if v == dest:
                    found = True
                    break
                q.append(v)
    res.explored = explored
    if not found:
        res.runtime_ms = (_time.perf_counter() - t0) * 1000
        return res
    path = [dest]
    while parent[path[-1]] is not None:
        path.append(parent[path[-1]])
    path.reverse()
    steps = _price_stop_path(net, path, cm)
    res.runtime_ms = (_time.perf_counter() - t0) * 1000
    return _finish(res, steps, cm)


def _price_stop_path(net: TransitNetwork, path: List[int], cm: CostModel):
    """Choose routes for each hop of a fixed stop path minimising cm-cost."""
    if len(path) == 1:
        return []
    # dp[i][rid] = (cost, prev_rid)
    dp: List[Dict[str, Tuple[float, Optional[str]]]] = []
    for i, (a, b) in enumerate(zip(path, path[1:])):
        layer = {}
        for rid, t, f in net.lookup[(a, b)]:
            ride_w = cm.weight(t, 0, f)
            rt = net.routes[rid]
            board_w = cm.weight(rt.wait_min, 0, rt.boarding_fee)
            if i == 0:
                layer[rid] = (board_w + ride_w, None)
            else:
                best = (float("inf"), None)
                for prid, (pc, _) in dp[-1].items():
                    c = pc + ride_w + (0.0 if prid == rid else
                                       cm.weight(rt.wait_min, 1, rt.boarding_fee))
                    if c < best[0]:
                        best = (c, prid)
                layer[rid] = (best[0], best[1])
        dp.append(layer)
    rid = min(dp[-1], key=lambda r: dp[-1][r][0])
    assign = [rid]
    for i in range(len(dp) - 1, 0, -1):
        rid = dp[i][rid][1]
        assign.append(rid)
    assign.reverse()

    steps = []
    cur: State = (path[0], None)
    prev_rid = None
    for (a, b), rid in zip(zip(path, path[1:]), assign):
        rt = net.routes[rid]
        if rid != prev_rid:
            nxt = (a, rid)
            steps.append(("board" if prev_rid is None else "transfer", cur, nxt,
                          rt.wait_min, rt.boarding_fee, 0 if prev_rid is None else 1))
            cur = nxt
        t, f = next((t, f) for r, t, f in net.lookup[(a, b)] if r == rid)
        nxt = (b, rid)
        steps.append(("ride", cur, nxt, t, f, 0))
        cur, prev_rid = nxt, rid
    return steps


# --------------------------------------------------------------------------
# Algorithm registry + measurement helpers
# --------------------------------------------------------------------------

ALGORITHMS: Dict[str, dict] = {
    "BFS":                         dict(algo="bfs"),
    "Dijkstra":                    dict(algo="dijkstra"),
    "A* (admissible)":             dict(algo="astar", kind="euclid", weight=1.0),
    "A* (0.5x, weak)":             dict(algo="astar", kind="euclid", weight=0.5),
    "A* (Manhattan)":              dict(algo="astar", kind="manhattan", weight=1.0),
    "A* (2x, inadmissible)":       dict(algo="astar", kind="euclid", weight=2.0),
}


def run(net: TransitNetwork, label: str, origin: int, dest: int,
        cm: CostModel) -> SearchResult:
    spec = ALGORITHMS[label]
    if spec["algo"] == "bfs":
        r = bfs(net, origin, dest, cm)
    elif spec["algo"] == "dijkstra":
        r = dijkstra(net, origin, dest, cm)
    else:
        h = make_heuristic(net, dest, cm, spec["kind"], spec["weight"])
        r = a_star(net, origin, dest, cm, h, name=label)
    r.algorithm = label
    return r


def measure(net, label, origin, dest, cm, repeats: int = 3,
            memory: bool = True) -> SearchResult:
    """Median runtime over ``repeats`` + one tracemalloc run for peak memory."""
    times = []
    r = None
    for _ in range(repeats):
        r = run(net, label, origin, dest, cm)
        times.append(r.runtime_ms)
    times.sort()
    r.runtime_ms = times[len(times) // 2]
    if memory:
        tracemalloc.start()
        tracemalloc.reset_peak()
        run(net, label, origin, dest, cm)
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        r.peak_mem_kb = peak / 1024.0
    return r
