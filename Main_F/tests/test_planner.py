import io, random, zipfile
import networkx as nx
import pytest
from transit import *
from transit.search import successors
from transit.analysis import consistency_report, admissibility_report

CMS = list(PRESETS.values())


@pytest.fixture(scope="module")
def net():
    return make_benchmark("medium", seed=7)


def pairs(n, k=12, seed=0):
    rng = random.Random(seed)
    return [tuple(rng.sample(range(n.n_stops), 2)) for _ in range(k)]


def test_astar_matches_dijkstra(net):
    for cm in CMS:
        for o, d in pairs(net):
            a = run(net, "Dijkstra", o, d, cm)
            b = run(net, "A* (admissible)", o, d, cm)
            assert a.found and b.found
            assert abs(a.objective - b.objective) < 1e-6


def test_matches_networkx_on_state_graph(net):
    cm = PRESETS["Balanced"]
    G = nx.DiGraph()
    for s in net.stops:
        for st in [(s, None)] + [(s, r) for r in net.routes_at[s]]:
            for ns, _, t, f, tr in successors(net, st):
                G.add_edge(st, ns, weight=cm.weight(t, tr, f))
    for o, d in pairs(net, 8, seed=3):
        best = min(nx.dijkstra_path_length(G, (o, None), (d, r))
                   for r in net.routes_at[d] if (d, r) in G)
        assert abs(best - run(net, "Dijkstra", o, d, cm).objective) < 1e-6


def test_heuristic_consistent_everywhere(net):
    for cm in CMS:
        for d in random.Random(1).sample(range(net.n_stops), 4):
            h = make_heuristic(net, d, cm, "euclid", 1.0)
            rep = consistency_report(net, d, cm, h)
            assert rep["violations"] == 0 and rep["h_goal"] == 0


def test_heuristic_admissible_sampled(net):
    cm = PRESETS["Balanced"]
    d = 17
    h = make_heuristic(net, d, cm, "euclid", 1.0)
    assert admissibility_report(net, d, cm, h, range(0, net.n_stops, 15))["violations"] == 0


def test_inadmissible_never_better_than_optimal(net):
    cm = PRESETS["Fastest"]
    for o, d in pairs(net, 15, seed=5):
        opt = run(net, "Dijkstra", o, d, cm).objective
        sub = run(net, "A* (2x, inadmissible)", o, d, cm).objective
        assert sub >= opt - 1e-6


def test_bfs_valid_and_not_better(net):
    cm = PRESETS["Fastest"]
    for o, d in pairs(net, 10, seed=9):
        b, dj = run(net, "BFS", o, d, cm), run(net, "Dijkstra", o, d, cm)
        assert b.found and b.stops[0] == o and b.stops[-1] == d
        for u, v in zip(b.stops, b.stops[1:]):
            assert v in net.adjacency[u]
        assert b.objective >= dj.objective - 1e-6


def test_origin_equals_destination(net):
    r = run(net, "A* (admissible)", 5, 5, PRESETS["Fastest"])
    assert r.found and r.total_time == 0 and r.stops == []


def test_itinerary_consistent(net):
    r = run(net, "Dijkstra", 0, 200, PRESETS["Balanced"])
    legs = r.legs(net)
    assert abs(sum(l["minutes"] for l in legs) - r.total_time) < 1e-6
    assert len(legs) - 1 == r.transfers


def test_gtfs_loader(tmp_path):
    files = {
        "stops.txt": "stop_id,stop_name,stop_lat,stop_lon\nA,Alpha,23.00,72.50\nB,Beta,23.01,72.51\nC,Gamma,23.02,72.52\nD,Delta,23.03,72.50\n",
        "routes.txt": "route_id,route_type\nR1,3\nR2,1\n",
        "trips.txt": "route_id,trip_id\nR1,t1\nR1,t2\nR2,t3\n",
        "stop_times.txt": ("trip_id,stop_sequence,stop_id,arrival_time,departure_time\n"
                           "t1,1,A,08:00:00,08:00:00\nt1,2,B,08:10:00,08:10:00\nt1,3,C,08:20:00,08:20:00\n"
                           "t2,1,A,09:00:00,09:00:00\nt2,2,B,09:10:00,09:10:00\nt2,3,C,09:20:00,09:20:00\n"
                           "t3,1,C,08:00:00,08:00:00\nt3,2,D,08:05:00,08:05:00\n"),
    }
    for k, v in files.items():
        (tmp_path / k).write_text(v)
    n = load_gtfs(tmp_path)
    assert n.n_stops == 4 and n.n_routes == 2
    r = run(n, "Dijkstra", 0, 3, PRESETS["Fastest"])
    assert r.found and r.transfers == 1 and r.stops == [0, 1, 2, 3]
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        for k, v in files.items():
            z.writestr(k, v)
    assert load_gtfs(buf.getvalue()).n_stops == 4
