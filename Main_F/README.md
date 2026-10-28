# Public Transport Route Planner

Graph-search route planner (BFS · Dijkstra · A*) on a route-aware transit network,
with heuristic analysis, benchmark suite and Streamlit app.

```
pip install -r requirements.txt
streamlit run app.py        # interactive prototype
pytest -q                   # 9 tests
python run_experiments.py   # regenerate results/ (≈1-2 min)
python make_report.py       # regenerate REPORT.md
```

| Path | Content |
|---|---|
| `transit/network.py` | graph model, synthetic generator, GTFS loader |
| `transit/search.py` | cost model, BFS, Dijkstra, A*, measurement helpers |
| `transit/heuristics.py` | admissible straight-line heuristic (+ variants) |
| `transit/analysis.py` | consistency / admissibility checkers |
| `app.py` | Streamlit UI (map, itinerary, algorithm comparison) |
| `REPORT.md`, `results/` | final report, CSVs, figures |

Python API:
```python
from transit import *
net = make_benchmark("medium")
r = run(net, "A* (admissible)", 3, 250, PRESETS["Balanced"])
print(r.total_time, r.total_fare, r.transfers, r.legs(net))
```
