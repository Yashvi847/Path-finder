"""
Transit network model.

The network is modelled as a weighted, route-aware graph G = (V, E):

    V : stops / stations (with planar coordinates in km)
    E : "ride" edges  (u -> v on route r)  carrying travel time (min) and fare.

Because a transfer between routes has its own time (waiting) and fare
(boarding fee), the search state is  s = (stop, route)  rather than just the
stop.  A route of ``None`` means "not on any vehicle yet" (initial state).

Two data sources are supported:
  * ``generate_network``  - controlled synthetic benchmark networks
  * ``load_gtfs``         - real GTFS feeds (directory or .zip)
"""
from __future__ import annotations

import io
import math
import random
import zipfile
from dataclasses import dataclass
from collections import defaultdict
from typing import Dict, List, Tuple, Optional

import numpy as np

# --------------------------------------------------------------------------
# Data classes
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Stop:
    id: int
    name: str
    x: float  # km
    y: float  # km


@dataclass
class Route:
    id: str
    mode: str                 # "bus" | "train" | ...
    stops: List[int]          # ordered stop ids
    headway_min: float        # average gap between vehicles
    boarding_fee: float       # fare paid every time this route is boarded
    rate_per_km: float        # fare per km travelled

    @property
    def wait_min(self) -> float:
        """Expected waiting time when boarding = half the headway."""
        return self.headway_min / 2.0


class TransitNetwork:
    """Container + fast lookup tables for a transit network."""

    def __init__(self, stops: List[Stop], routes: List[Route],
                 ride_edges: List[Tuple[str, int, int, float, float]],
                 name: str = "network"):
        """
        ride_edges: list of (route_id, u, v, minutes, fare)
        """
        self.name = name
        self.stops: Dict[int, Stop] = {s.id: s for s in stops}
        self.routes: Dict[str, Route] = {r.id: r for r in routes}

        # rides[(u, rid)] -> [(v, minutes, fare), ...]
        self.rides: Dict[Tuple[int, str], List[Tuple[int, float, float]]] = defaultdict(list)
        # lookup[(u, v)] -> [(rid, minutes, fare), ...]
        self.lookup: Dict[Tuple[int, int], List[Tuple[str, float, float]]] = defaultdict(list)
        self.routes_at: Dict[int, List[str]] = defaultdict(list)
        self.adjacency: Dict[int, set] = defaultdict(set)

        for rid, u, v, t, f in ride_edges:
            self.rides[(u, rid)].append((v, t, f))
            self.lookup[(u, v)].append((rid, t, f))
            self.adjacency[u].add(v)
            if rid not in self.routes_at[u]:
                self.routes_at[u].append(rid)
            if rid not in self.routes_at[v]:
                self.routes_at[v].append(rid)
        self._edges = ride_edges
        self._bounds: Optional[Tuple[float, float]] = None

    # ---- basic geometry ---------------------------------------------------
    def dist(self, a: int, b: int) -> float:
        sa, sb = self.stops[a], self.stops[b]
        return math.hypot(sa.x - sb.x, sa.y - sb.y)

    def manhattan(self, a: int, b: int) -> float:
        sa, sb = self.stops[a], self.stops[b]
        return abs(sa.x - sb.x) + abs(sa.y - sb.y)

    # ---- data-derived bounds used by the admissible heuristic -------------
    def bounds(self) -> Tuple[float, float]:
        """
        Returns (v_max_kmh, min_fare_per_km) *measured from the data*:

          v_max_kmh     = max over ride edges of  euclid(u,v) / time(u,v)
          min_fare_per_km = min over ride edges of  fare(u,v) / euclid(u,v)

        Using straight-line distance in the numerator/denominator guarantees
        that  time(e) >= euclid(e)/v_max  and  fare(e) >= rate_min*euclid(e)
        for every edge, which (with the triangle inequality) makes the
        straight-line heuristic admissible *by construction*.
        """
        if self._bounds is None:
            vmax, rmin = 0.0, float("inf")
            for _, u, v, t, f in self._edges:
                d = self.dist(u, v)
                if d <= 1e-9:
                    continue
                if t > 0:
                    vmax = max(vmax, d / (t / 60.0))
                rmin = min(rmin, f / d)
            self._bounds = (vmax or 1.0, 0.0 if rmin == float("inf") else rmin)
        return self._bounds

    # ---- summary ----------------------------------------------------------
    @property
    def n_stops(self) -> int:
        return len(self.stops)

    @property
    def n_routes(self) -> int:
        return len(self.routes)

    @property
    def n_edges(self) -> int:
        return len(self._edges)

    @property
    def n_states(self) -> int:
        return sum(len(v) for v in self.routes_at.values()) + len(self.stops)

    def summary(self) -> dict:
        modes = defaultdict(int)
        for r in self.routes.values():
            modes[r.mode] += 1
        return {
            "name": self.name, "stops": self.n_stops, "routes": self.n_routes,
            "ride_edges": self.n_edges, "search_states": self.n_states,
            "modes": dict(modes),
            "avg_routes_per_stop": round(
                sum(len(v) for v in self.routes_at.values()) / max(1, self.n_stops), 2),
        }

    def to_networkx(self):
        """Stop-level multigraph (for visualisation / cross-validation)."""
        import networkx as nx
        G = nx.MultiDiGraph(name=self.name)
        for s in self.stops.values():
            G.add_node(s.id, name=s.name, x=s.x, y=s.y)
        for rid, u, v, t, f in self._edges:
            G.add_edge(u, v, key=rid, route=rid, time=t, fare=f,
                       mode=self.routes[rid].mode)
        return G


# --------------------------------------------------------------------------
# Synthetic benchmark generator
# --------------------------------------------------------------------------

BUS = dict(mode="bus", speed=(18.0, 28.0), headway=(8.0, 15.0),
           boarding_fee=5.0, rate=1.0)
TRAIN = dict(mode="train", speed=(45.0, 60.0), headway=(10.0, 20.0),
             boarding_fee=10.0, rate=1.5)
DWELL_MIN = 0.5


def _angle_diff(a: float, b: float) -> float:
    d = (a - b + math.pi) % (2 * math.pi) - math.pi
    return abs(d)


def _knn(coords: np.ndarray, ids: np.ndarray, k: int) -> Dict[int, List[int]]:
    """k nearest neighbours (by id) restricted to the subset ``ids``."""
    sub = coords[ids]
    d2 = ((sub[:, None, :] - sub[None, :, :]) ** 2).sum(-1)
    np.fill_diagonal(d2, np.inf)
    k = min(k, len(ids) - 1)
    out = {}
    for i, sid in enumerate(ids):
        idx = np.argpartition(d2[i], k - 1)[:k]
        idx = idx[np.argsort(d2[i][idx])]
        out[int(sid)] = [int(ids[j]) for j in idx]
    return out


def _walk(start, length, knn, coords, rng) -> List[int]:
    """Directionally persistent walk -> a plausible line shape."""
    heading = rng.uniform(-math.pi, math.pi)
    path, seen, cur = [start], {start}, start
    for _ in range(length - 1):
        best = None
        for nb in knn[cur]:
            if nb in seen:
                continue
            dx, dy = coords[nb] - coords[cur]
            dev = _angle_diff(math.atan2(dy, dx), heading)
            if dev > math.pi / 1.8:
                continue
            score = dev + rng.random() * 0.4
            if best is None or score < best[0]:
                best = (score, nb)
        if best is None:
            break
        nb = best[1]
        dx, dy = coords[nb] - coords[cur]
        heading = 0.7 * heading + 0.3 * math.atan2(dy, dx) + rng.gauss(0, 0.15)
        path.append(nb)
        seen.add(nb)
        cur = nb
    return path


class _DSU:
    def __init__(self, n):
        self.p = list(range(n))

    def find(self, a):
        while self.p[a] != a:
            self.p[a] = self.p[self.p[a]]
            a = self.p[a]
        return a

    def union(self, a, b):
        self.p[self.find(a)] = self.find(b)


def generate_network(n_stops: int, n_bus_routes: int, n_train_routes: int,
                     seed: int = 0, name: Optional[str] = None) -> TransitNetwork:
    """
    Controlled synthetic network.

    * Stops are uniform random points in a square whose side grows with
      sqrt(n) (constant density ~0.8 stops / km^2).
    * Bus lines are directional random walks over nearest neighbours.
    * Train lines are walks over a sparse subset of "hub" stops with long hops.
    * Feeder lines are added so every stop is served and the graph is
      connected (so every origin-destination pair has a solution).
    * Every ride edge has a circuity factor in [1.05, 1.30]; therefore the
      network never beats straight-line geometry (realistic).
    """
    rng = random.Random(seed)
    area = max(6.0, math.sqrt(n_stops) * 1.1)
    coords = np.array([[rng.uniform(0, area), rng.uniform(0, area)]
                       for _ in range(n_stops)])
    all_ids = np.arange(n_stops)
    knn_bus = _knn(coords, all_ids, 6)

    lines: List[Tuple[str, List[int]]] = []   # (mode, stop list)

    for _ in range(n_bus_routes):
        p = _walk(rng.randrange(n_stops), rng.randint(8, 22), knn_bus, coords, rng)
        if len(p) >= 4:
            lines.append(("bus", p))

    if n_train_routes > 0:
        n_hub = max(6, n_stops // 6)
        hubs = np.array(sorted(rng.sample(range(n_stops), n_hub)))
        knn_hub = _knn(coords, hubs, 5)
        for _ in range(n_train_routes):
            p = _walk(int(rng.choice(hubs)), rng.randint(5, 12), knn_hub, coords, rng)
            if len(p) >= 3:
                lines.append(("train", p))

    # --- feeders: serve every stop, then connect components ---------------
    covered = {s for _, p in lines for s in p}
    for s in range(n_stops):
        if s not in covered:
            cand = sorted(covered or {0}, key=lambda c: np.hypot(*(coords[s] - coords[c])))
            tgt = cand[0] if cand else (s + 1) % n_stops
            lines.append(("bus", [s, tgt]))
            covered.add(s)

    def components():
        dsu = _DSU(n_stops)
        for _, p in lines:
            for a, b in zip(p, p[1:]):
                dsu.union(a, b)
        comp = defaultdict(list)
        for s in range(n_stops):
            comp[dsu.find(s)].append(s)
        return sorted(comp.values(), key=len, reverse=True)

    comps = components()
    while len(comps) > 1:
        main = np.array(comps[0])
        for other in comps[1:]:
            oth = np.array(other)
            d2 = ((coords[oth][:, None, :] - coords[main][None, :, :]) ** 2).sum(-1)
            i, j = np.unravel_index(np.argmin(d2), d2.shape)
            lines.append(("bus", [int(oth[i]), int(main[j])]))
        comps = components()

    # --- materialise routes & ride edges ----------------------------------
    stops = [Stop(i, f"S{i:04d}", float(coords[i][0]), float(coords[i][1]))
             for i in range(n_stops)]
    routes: List[Route] = []
    edges: List[Tuple[str, int, int, float, float]] = []
    counters = defaultdict(int)
    for mode, path in lines:
        spec = BUS if mode == "bus" else TRAIN
        counters[mode] += 1
        rid = f"{'B' if mode == 'bus' else 'T'}{counters[mode]:03d}"
        speed = rng.uniform(*spec["speed"])
        routes.append(Route(rid, mode, path, rng.uniform(*spec["headway"]),
                            spec["boarding_fee"], spec["rate"]))
        for a, b in zip(path, path[1:]):
            km = float(np.hypot(*(coords[a] - coords[b]))) * rng.uniform(1.05, 1.30)
            t = km / speed * 60.0 + DWELL_MIN
            f = km * spec["rate"]
            edges.append((rid, a, b, t, f))
            edges.append((rid, b, a, t, f))   # lines run in both directions

    return TransitNetwork(stops, routes, edges,
                          name=name or f"synthetic-{n_stops}")


SIZES = {
    "small":  dict(n_stops=40,   n_bus_routes=10,  n_train_routes=2),
    "medium": dict(n_stops=300,  n_bus_routes=75,  n_train_routes=8),
    "large":  dict(n_stops=1500, n_bus_routes=380, n_train_routes=30),
}


def make_benchmark(size: str, seed: int = 42) -> TransitNetwork:
    return generate_network(**SIZES[size], seed=seed, name=f"{size}-{seed}")


# --------------------------------------------------------------------------
# GTFS loader (real-world data)
# --------------------------------------------------------------------------

DEFAULT_FARES = {
    # GTFS route_type -> (mode, boarding_fee, rate_per_km)
    0: ("tram", 8.0, 1.2), 1: ("metro", 10.0, 1.5), 2: ("train", 10.0, 1.5),
    3: ("bus", 5.0, 1.0), 4: ("ferry", 8.0, 1.5),
}


def _hms_to_min(s: str) -> float:
    h, m, sec = (int(x) for x in str(s).strip().split(":"))
    return h * 60 + m + sec / 60.0


def load_gtfs(source, fares: Optional[dict] = None, service_hours: float = 16.0,
              max_patterns: Optional[int] = None) -> TransitNetwork:
    """
    Load a GTFS feed (directory path, .zip path, or file-like object).

    Required files: stops.txt, routes.txt, trips.txt, stop_times.txt.
    Each distinct (route, stop sequence) becomes one directed Route in the
    model.  Segment time is taken from the first trip of the pattern, headway
    from the number of trips (``service_hours / n_trips``).  GTFS has no
    fares, so fares come from a per-mode table (``DEFAULT_FARES``).
    """
    import os
    import pandas as pd

    fares = {**DEFAULT_FARES, **(fares or {})}

    def read(name):
        if isinstance(source, (str, os.PathLike)) and os.path.isdir(source):
            return pd.read_csv(os.path.join(source, name), dtype=str)
        z = zipfile.ZipFile(source if not isinstance(source, bytes) else io.BytesIO(source))
        with z.open(name) as fh:
            return pd.read_csv(fh, dtype=str)

    stops_df = read("stops.txt")
    routes_df = read("routes.txt")
    trips_df = read("trips.txt")
    st_df = read("stop_times.txt")

    lat0 = stops_df["stop_lat"].astype(float).mean()
    lon0 = stops_df["stop_lon"].astype(float).mean()
    kx = 111.32 * math.cos(math.radians(lat0))
    ky = 110.574
    sid_map, stops = {}, []
    for i, row in enumerate(stops_df.itertuples(index=False)):
        sid_map[row.stop_id] = i
        stops.append(Stop(i, getattr(row, "stop_name", row.stop_id),
                          (float(row.stop_lon) - lon0) * kx,
                          (float(row.stop_lat) - lat0) * ky))

    rtype = dict(zip(routes_df["route_id"],
                     routes_df.get("route_type", pd.Series(["3"] * len(routes_df))).astype(int)))
    trip_route = dict(zip(trips_df["trip_id"], trips_df["route_id"]))

    st_df["stop_sequence"] = st_df["stop_sequence"].astype(int)
    st_df = st_df.sort_values(["trip_id", "stop_sequence"])

    patterns: Dict[Tuple[str, Tuple[int, ...]], dict] = {}
    for trip_id, grp in st_df.groupby("trip_id", sort=False):
        if trip_id not in trip_route:
            continue
        seq = tuple(sid_map[s] for s in grp["stop_id"] if s in sid_map)
        if len(seq) < 2:
            continue
        key = (trip_route[trip_id], seq)
        if key not in patterns:
            times = [_hms_to_min(t) for t in grp["arrival_time"]]
            patterns[key] = dict(times=times, trips=1)
        else:
            patterns[key]["trips"] += 1

    items = list(patterns.items())
    if max_patterns:
        items = items[:max_patterns]

    routes, edges = [], []
    for k, ((rid0, seq), info) in enumerate(items):
        mode, fee, rate = fares.get(rtype.get(rid0, 3), fares[3])
        rid = f"{rid0}#{k}"
        headway = max(2.0, service_hours * 60.0 / info["trips"])
        routes.append(Route(rid, mode, list(seq), headway, fee, rate))
        for a, b, ta, tb in zip(seq, seq[1:], info["times"], info["times"][1:]):
            t = max(0.5, tb - ta)
            km = math.hypot(stops[a].x - stops[b].x, stops[a].y - stops[b].y)
            edges.append((rid, a, b, t, km * rate))
    return TransitNetwork(stops, routes, edges, name="gtfs")
