"""Interactive Public Transport Route Planner.   Run:  streamlit run app.py"""
import random
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from transit import (make_benchmark, load_gtfs, PRESETS, CostModel, ALGORITHMS,
                     run, measure, SIZES)

st.set_page_config(page_title="Public Transport Route Planner", page_icon="🚌", layout="wide")


# ----------------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------------
@st.cache_resource(show_spinner="Building network...")
def synthetic(size: str, seed: int):
    return make_benchmark(size, seed)


@st.cache_resource(show_spinner="Parsing GTFS feed...")
def gtfs(raw: bytes):
    return load_gtfs(raw)


def fmt_stop(net, sid):
    return net.stops[sid].name


# ----------------------------------------------------------------------------
# Sidebar
# ----------------------------------------------------------------------------
st.sidebar.title("🚌 Route Planner")
src = st.sidebar.radio("Network", ["Synthetic benchmark", "Upload GTFS (.zip)"])
if src == "Synthetic benchmark":
    size = st.sidebar.select_slider("Network size", list(SIZES), value="medium")
    seed = st.sidebar.number_input("Random seed", 0, 9999, 42)
    net = synthetic(size, int(seed))
else:
    up = st.sidebar.file_uploader("GTFS feed", type="zip")
    if up is None:
        st.info("Upload a GTFS .zip (stops, routes, trips, stop_times) or switch to the synthetic benchmark.")
        st.stop()
    net = gtfs(up.getvalue())

ids = sorted(net.stops)
if "o" not in st.session_state or st.session_state.get("net") != net.name + str(net.n_stops):
    rng = random.Random(1)
    st.session_state.o, st.session_state.d = rng.sample(ids, 2)
    st.session_state.net = net.name + str(net.n_stops)

if st.sidebar.button("🎲 Random origin / destination"):
    st.session_state.o, st.session_state.d = random.sample(ids, 2)

origin = st.sidebar.selectbox("Origin", ids, index=ids.index(st.session_state.o),
                              format_func=lambda i: fmt_stop(net, i), key="o_sel")
dest = st.sidebar.selectbox("Destination", ids, index=ids.index(st.session_state.d),
                            format_func=lambda i: fmt_stop(net, i), key="d_sel")

crit = st.sidebar.selectbox("Optimise for", list(PRESETS) + ["Custom weights"], index=3)
if crit == "Custom weights":
    a = st.sidebar.slider("α  time weight (per minute)", 0.0, 2.0, 1.0, 0.05)
    b = st.sidebar.slider("β  transfer penalty", 0.0, 30.0, 5.0, 0.5)
    c = st.sidebar.slider("γ  fare weight (per currency unit)", 0.0, 3.0, 0.5, 0.05)
    cm = CostModel(a, b, c, "Custom")
else:
    cm = PRESETS[crit]
st.sidebar.caption(f"C = {cm.alpha}·T + {cm.beta}·R + {cm.gamma}·K")

algo = st.sidebar.selectbox("Algorithm", list(ALGORITHMS), index=2)
show_explored = st.sidebar.checkbox("Show explored stops", True)

# ----------------------------------------------------------------------------
# Header + tabs
# ----------------------------------------------------------------------------
st.title("Public Transport Route Planner")
st.caption("Graph search (BFS · Dijkstra · A*) on a route-aware transit network. "
           "State = (stop, route); step cost = α·time + β·transfers + γ·fare.")
tab_plan, tab_cmp, tab_net = st.tabs(["Plan a journey", "Compare algorithms", "Network"])


def draw_map(net, res, show_explored):
    fig = go.Figure()
    xs, ys = [], []
    for (u, rid), lst in net.rides.items():
        for (v, _, _) in lst:
            if u < v:
                xs += [net.stops[u].x, net.stops[v].x, None]
                ys += [net.stops[u].y, net.stops[v].y, None]
    fig.add_trace(go.Scattergl(x=xs, y=ys, mode="lines", line=dict(width=0.6, color="#c8ccd2"),
                               hoverinfo="skip", name="network"))
    if show_explored and res.explored:
        fig.add_trace(go.Scattergl(
            x=[net.stops[s].x for s in res.explored], y=[net.stops[s].y for s in res.explored],
            mode="markers", marker=dict(size=5, color="#f6b26b", opacity=0.8),
            name=f"explored ({len(res.explored)})",
            text=[net.stops[s].name for s in res.explored], hoverinfo="text"))
    palette = {"bus": "#1f77b4", "train": "#d62728", "tram": "#2ca02c", "metro": "#9467bd"}
    seen = set()
    for step in res.steps:
        if step[0] != "ride":
            continue
        (u, _), (v, rid) = step[1], step[2]
        mode = net.routes[rid].mode
        fig.add_trace(go.Scatter(
            x=[net.stops[u].x, net.stops[v].x], y=[net.stops[u].y, net.stops[v].y],
            mode="lines", line=dict(width=5, color=palette.get(mode, "#333")),
            name=mode, legendgroup=mode, showlegend=mode not in seen, hoverinfo="skip"))
        seen.add(mode)
    for sid, label, col in [(res.stops[0], "Origin", "#2ca02c"), (res.stops[-1], "Destination", "#000000")]:
        fig.add_trace(go.Scatter(x=[net.stops[sid].x], y=[net.stops[sid].y], mode="markers+text",
                                 marker=dict(size=14, color=col, symbol="star"),
                                 text=[label], textposition="top center", name=label))
    fig.update_yaxes(scaleanchor="x", scaleratio=1, title="km")
    fig.update_xaxes(title="km")
    fig.update_layout(height=560, margin=dict(l=10, r=10, t=10, b=10),
                      legend=dict(orientation="h", y=1.02))
    return fig


with tab_plan:
    if origin == dest:
        st.warning("Choose two different stops.")
    else:
        res = run(net, algo, origin, dest, cm)
        if not res.found:
            st.error("No route found between these stops.")
        else:
            k = st.columns(5)
            k[0].metric("Travel time", f"{res.total_time:.1f} min")
            k[1].metric("Fare", f"{res.total_fare:.2f}")
            k[2].metric("Transfers", res.transfers)
            k[3].metric("Stops visited", len(res.stops))
            k[4].metric("Objective C", f"{res.objective:.2f}")
            left, right = st.columns([3, 2])
            with left:
                st.plotly_chart(draw_map(net, res, show_explored), width="stretch")
            with right:
                st.subheader("Itinerary")
                for i, leg in enumerate(res.legs(net), 1):
                    icon = "🚆" if leg["mode"] in ("train", "metro") else "🚌"
                    verb = "Transfer to" if leg["is_transfer"] else "Board"
                    st.markdown(
                        f"**{i}. {verb} {icon} {leg['route']}** at **{fmt_stop(net, leg['from_stop'])}**  \n"
                        f"→ ride {leg['hops']} stop(s) to **{fmt_stop(net, leg['to_stop'])}**  \n"
                        f"⏱ {leg['minutes']:.1f} min (wait {leg['wait']:.1f}) · 💰 {leg['fare']:.2f}")
                st.success(f"Arrive at **{fmt_stop(net, dest)}**")
                st.subheader("Search statistics")
                st.table(pd.DataFrame({"Algorithm": [res.algorithm],
                                       "Generated": [res.generated], "Expanded": [res.expanded],
                                       "Max frontier": [res.max_frontier],
                                       "Runtime (ms)": [round(res.runtime_ms, 2)]}).set_index("Algorithm"))

with tab_cmp:
    st.write("Runs every algorithm on the selected origin / destination and cost model.")
    if origin != dest and st.button("Run comparison", type="primary"):
        rows = []
        opt = run(net, "Dijkstra", origin, dest, cm).objective
        for lab in ALGORITHMS:
            r = measure(net, lab, origin, dest, cm, repeats=3, memory=True)
            rows.append({"Algorithm": lab, "Generated": r.generated, "Expanded": r.expanded,
                         "Runtime (ms)": round(r.runtime_ms, 3), "Peak mem (KiB)": round(r.peak_mem_kb, 1),
                         "Time (min)": round(r.total_time, 1), "Fare": round(r.total_fare, 2),
                         "Transfers": r.transfers, "Objective": round(r.objective, 2),
                         "Optimal": "✅" if abs(r.objective - opt) < 1e-6 else f"❌ +{100*(r.objective-opt)/opt:.1f}%"})
        df = pd.DataFrame(rows).set_index("Algorithm")
        st.dataframe(df, width="stretch")
        st.bar_chart(df["Expanded"])

with tab_net:
    st.json(net.summary())
    st.write("Bounds used by the heuristic (measured from the data):")
    vmax, cmin = net.bounds()
    st.write(f"v_max = **{vmax:.1f} km/h**, cheapest fare = **{cmin:.2f} / km**")
