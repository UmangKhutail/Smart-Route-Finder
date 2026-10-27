
import heapq
import math
import re
import time
from collections import defaultdict

import folium
import requests
import streamlit as st
from streamlit_folium import st_folium

# ----------------------------------------------------------------------------
# OpenStreetMap helpers (free, no API key)
# ----------------------------------------------------------------------------
NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"
OSRM_URL = "https://router.project-osrm.org/route/v1/driving/{o_lng},{o_lat};{d_lng},{d_lat}"
HEADERS = {"User-Agent": "DAA-Dijkstra-Route-Finder/1.0 (college project)"}


def decode_polyline(encoded):
    """Decode an encoded polyline string (precision 5) into [(lat, lng), ...]."""
    coords, index, lat, lng = [], 0, 0, 0
    while index < len(encoded):
        for is_lat in (True, False):
            shift = result = 0
            while True:
                b = ord(encoded[index]) - 63
                index += 1
                result |= (b & 0x1F) << shift
                shift += 5
                if b < 0x20:
                    break
            delta = ~(result >> 1) if result & 1 else result >> 1
            if is_lat:
                lat += delta
            else:
                lng += delta
        coords.append((lat / 1e5, lng / 1e5))
    return coords


def geocode(place):
    r = requests.get(NOMINATIM_URL, params={"q": place, "format": "json", "limit": 1},
                     headers=HEADERS, timeout=20).json()
    if not r:
        raise RuntimeError(f"Place not found: '{place}'. Try a more specific name (add city / state).")
    return float(r[0]["lat"]), float(r[0]["lon"]), r[0]["display_name"]


@st.cache_data(show_spinner=False, ttl=600)
def fetch_routes(origin, destination):
    o_lat, o_lng, o_addr = geocode(origin)
    time.sleep(1)  # Nominatim fair-use policy: max 1 request / second
    d_lat, d_lng, d_addr = geocode(destination)
    base = "https://router.project-osrm.org/route/v1/driving/"
    params = {"alternatives": "3", "steps": "true", "geometries": "polyline", "overview": "false"}

    def osrm(coords):
        path = ";".join(f"{lng},{lat}" for lat, lng in coords)
        try:
            r = requests.get(base + path, params=params, headers=HEADERS, timeout=40).json()
            time.sleep(0.3)
            return r["routes"] if r.get("code") == "Ok" else []
        except Exception:
            return []

    routes = osrm([(o_lat, o_lng), (d_lat, d_lng)])
    if not routes:
        raise RuntimeError("Routing failed. OSRM server busy ya route nahi mila, thodi der baad try karo.")

    # --- Split the main route into hubs and collect alternatives for every segment ---
    main = routes[0]
    path_pts = [p for leg in main["legs"] for st_ in leg["steps"] for p in decode_polyline(st_["geometry"])]
    total_km = main["distance"] / 1000
    n_seg = max(2, min(5, round(total_km / 80)))
    idx = [int(i * (len(path_pts) - 1) / n_seg) for i in range(n_seg + 1)]
    hubs = [(o_lat, o_lng)] + [path_pts[i] for i in idx[1:-1]] + [(d_lat, d_lng)]

    for h1, h2 in zip(hubs[:-1], hubs[1:]):
        routes += osrm([h1, h2])  # direct + OSRM alternatives for this segment
        dy, dx = h2[0] - h1[0], h2[1] - h1[1]
        mid = ((h1[0] + h2[0]) / 2, (h1[1] + h2[1]) / 2)
        for side in (-1, 1):  # side detours -> roads through towns / old highways
            via = (mid[0] + side * (-dx) * 0.15, mid[1] + side * dy * 0.15)
            routes += osrm([h1, via, h2])
    return routes, (o_lat, o_lng, o_addr), (d_lat, d_lng, d_addr)


# ----------------------------------------------------------------------------
# Graph construction
# ----------------------------------------------------------------------------
HIGHWAY_WORDS = re.compile(r"\b(NH|SH|Expy|Expressway|Highway|Hwy|Freeway|Motorway|Bypass|Ring Road|Flyover|Marg)\b", re.I)
NON_TURN = {"depart", "arrive", "continue", "new name"}


def node_id(lat, lng):
    # ~11 m grid: steps of different routes that meet at the same junction merge into one node
    return (round(lat, 4), round(lng, 4))


def build_graph(routes):
    """Directed graph: node -> list of edges. Each OSRM 'step' is one edge."""
    graph = defaultdict(list)
    for ri, route in enumerate(routes):
        for leg in route["legs"]:
            for step in leg["steps"]:
                poly = decode_polyline(step["geometry"])
                if len(poly) < 2 or step["distance"] <= 0:
                    continue
                u, v = node_id(*poly[0]), node_id(*poly[-1])
                if u == v:
                    continue
                name = step.get("name") or "unnamed road"
                ref = step.get("ref", "")
                man = step.get("maneuver", {})
                mtype, mod = man.get("type", ""), man.get("modifier", "")
                text = f"{mtype} {mod} onto {name} {ref}".replace("  ", " ").strip()
                graph[u].append({
                    "to": v,
                    "dist_m": step["distance"],
                    "time_s": max(step["duration"], 1),
                    "text": text,
                    "is_turn": mtype not in NON_TURN,
                    "is_highway": bool(HIGHWAY_WORDS.search(name + " " + ref)) or bool(re.match(r"^(NH|SH)", ref, re.I)),
                    "poly": poly,
                    "route": ri,
                })
                graph.setdefault(v, [])
    return graph


# ----------------------------------------------------------------------------
# Edge-weight functions (one per button)
# ----------------------------------------------------------------------------
def speed_kmph(e):
    return max(5.0, min(120.0, (e["dist_m"] / 1000) / (e["time_s"] / 3600)))


def fuel_litres(e, mileage_kmpl):
    """U-shaped consumption curve: poor at crawling speeds and at very high speeds."""
    def raw(v):
        return 150 / v + 0.0007 * v * v + 1.0

    scale = (100.0 / mileage_kmpl) / raw(60)  # calibrate: user's mileage at 60 km/h
    v = speed_kmph(e)
    return (e["dist_m"] / 1000) * scale * raw(v) / 100.0


def make_weight_fn(mode, mileage=None):
    if mode == "dist":      # 📏 Shortest Distance
        return lambda e: e["dist_m"] / 1000
    if mode == "turns":     # 🧭 Fewest Turns (easy driving); tiny distance term breaks ties
        return lambda e: (1000 if e["is_turn"] else 0) + e["dist_m"] / 1000
    if mode == "local":     # 🌄 Avoid Highways (local / scenic roads)
        def w(e):
            fast = e["is_highway"] or speed_kmph(e) >= 75
            return (e["dist_m"] / 1000) * (10 if fast else 1) + (0.3 if e["is_turn"] else 0)
        return w
    raise ValueError(mode)


# ----------------------------------------------------------------------------
# Dijkstra's algorithm  (min-heap implementation)
# ----------------------------------------------------------------------------
def dijkstra(graph, src, dst, weight_fn):
    dist = {n: math.inf for n in graph}
    prev = {}  # node -> (previous node, edge used)
    dist[src] = 0.0
    heap = [(0.0, src)]
    visited = set()
    relaxations = 0

    while heap:
        d, u = heapq.heappop(heap)
        if u in visited:
            continue
        visited.add(u)
        if u == dst:
            break
        for e in graph[u]:
            v = e["to"]
            nd = d + weight_fn(e)
            relaxations += 1
            if nd < dist.get(v, math.inf):
                dist[v] = nd
                prev[v] = (u, e)
                heapq.heappush(heap, (nd, v))

    if dist.get(dst, math.inf) == math.inf:
        return None
    edges, cur = [], dst
    while cur != src:
        u, e = prev[cur]
        edges.append(e)
        cur = u
    edges.reverse()
    return {"cost": dist[dst], "edges": edges, "visited": len(visited), "relaxations": relaxations}


def nearest_node(graph, lat, lng):
    return min(graph, key=lambda n: (n[0] - lat) ** 2 + (n[1] - lng) ** 2)


def summarize(result, mileage, price):
    edges = result["edges"]
    dist_km = sum(e["dist_m"] for e in edges) / 1000
    time_min = sum(e["time_s"] for e in edges) / 60
    fuel = sum(fuel_litres(e, mileage) for e in edges)
    hw = sum(e["dist_m"] for e in edges if e["is_highway"]) / 1000
    return {
        "dist_km": dist_km, "time_min": time_min, "fuel_l": fuel,
        "cost_rs": fuel * price, "hw_km": hw,
        "turns": sum(1 for e in edges if e["is_turn"]),
    }


# ----------------------------------------------------------------------------
# Streamlit UI
# ----------------------------------------------------------------------------
st.set_page_config(page_title="DAA - Dijkstra Route Finder", page_icon="🗺️", layout="wide")
st.title("🗺️ Smart Route Finder - Dijkstra + OpenStreetMap")
st.caption("DAA Project · Single-source shortest path with 3 different edge-weight strategies")

with st.sidebar:
    st.header("⚙️ Settings")
    st.success("✅ Free mode: OpenStreetMap + OSRM (no API key needed)")
    origin = st.text_input("From", "Ludhiana Railway Station, Punjab")
    destination = st.text_input("To", "Chandigarh Sector 17, Chandigarh")
    mileage = st.number_input("Vehicle mileage (km/l)", 5.0, 50.0, 18.0, 0.5)
    price = st.number_input("Fuel price (₹/litre)", 50.0, 200.0, 95.0, 1.0)

st.subheader("Choose what to optimise")
c1, c2, c3 = st.columns(3)
mode = None
if c1.button("📏 Shortest Distance", use_container_width=True):
    mode = "dist"
if c2.button("🧭 Fewest Turns (Easy Drive)", use_container_width=True):
    mode = "turns"
if c3.button("🌄 Avoid Highways (Local Roads)", use_container_width=True):
    mode = "local"

if mode:
    try:
        with st.spinner("Fetching roads from OpenStreetMap (The first time, it will take around 20–40 seconds.)..."):
            routes, o, d = fetch_routes(origin, destination)
        graph = build_graph(routes)
        src = nearest_node(graph, o[0], o[1])
        dst = nearest_node(graph, d[0], d[1])

        results = {}
        for m in ("dist", "turns", "local"):
            res = dijkstra(graph, src, dst, make_weight_fn(m, mileage))
            if res is None:
                raise RuntimeError("No path found in graph.")
            results[m] = (res, summarize(res, mileage, price))
        st.session_state["last"] = (mode, results, o, d, len(graph), sum(len(v) for v in graph.values()))
    except Exception as ex:
        st.error(str(ex))
        st.stop()

if "last" in st.session_state:
    mode, results, o, d, V, E = st.session_state["last"]
    titles = {"dist": "📏 Shortest Distance", "turns": "🧭 Fewest Turns (Easy Drive)", "local": "🌄 Avoid Highways (Local Roads)"}
    colors = {"dist": "blue", "turns": "green", "local": "red"}
    res, s = results[mode]

    st.success(f"**{titles[mode]}** route  |  {o[2]}  ➜  {d[2]}")
    sig = {k: tuple(id(e) for e in results[k][0]["edges"]) for k in results}
    same = [titles[k] for k in results if k != mode and sig[k] == sig[mode]]
    if same:
        st.info("ℹ️ Is route ke liye ye mode **" + ", ".join(same) + "** jaisa hi best raasta deta hai "
                "(graph me isse behtar alternative nahi mila). Dusre mode ka alag raasta dekhne ke liye uska button dabao "
                "ya aisa route try karo jisme kai raaste hon.")
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Distance", f"{s['dist_km']:.1f} km")
    m2.metric("Time", f"{s['time_min']:.0f} min")
    m3.metric("Fuel", f"{s['fuel_l']:.2f} L")
    m4.metric("Fuel cost", f"₹{s['cost_rs']:.0f}")
    m5.metric("Highway share", f"{(s['hw_km'] / s['dist_km'] * 100 if s['dist_km'] else 0):.0f}%")

    fmap = folium.Map(location=[o[0], o[1]], zoom_start=10, tiles="OpenStreetMap")
    pts = [p for e in res["edges"] for p in e["poly"]]
    folium.PolyLine(pts, color=colors[mode], weight=6, opacity=0.85).add_to(fmap)
    folium.Marker([o[0], o[1]], tooltip="Start", icon=folium.Icon(color="green", icon="play")).add_to(fmap)
    folium.Marker([d[0], d[1]], tooltip="Destination", icon=folium.Icon(color="red", icon="flag")).add_to(fmap)
    fmap.fit_bounds(pts)
    st_folium(fmap, height=500, use_container_width=True, returned_objects=[])

    st.subheader("📊 Comparison of all three strategies")
    st.table({
        "Strategy": [titles[k] for k in results],
        "Distance (km)": [f"{results[k][1]['dist_km']:.1f}" for k in results],
        "Time (min)": [f"{results[k][1]['time_min']:.0f}" for k in results],
        "Fuel (L)": [f"{results[k][1]['fuel_l']:.2f}" for k in results],
        "Turns": [results[k][1]["turns"] for k in results],
        "Highway km": [f"{results[k][1]['hw_km']:.1f}" for k in results],
    })

    with st.expander("🧠 Algorithm details (for viva / report)"):
        st.markdown(f"""
**Graph built from OpenStreetMap (OSRM):** V = {V} junction nodes, E = {E} road-segment edges.

**Dijkstra run ({titles[mode]}):** {res['visited']} nodes settled, {res['relaxations']} edge relaxations.

| Mode | Edge weight w(u,v) |
|---|---|
| Shortest Distance | road length of the segment in km |
| Fewest Turns | 1000 per turn/maneuver + length in km (tie-break) → fewest turns, simplest drive |
| Avoid Highways | length × 10 if the segment is a highway / fast road (≥ 75 km/h), else length × 1 → local & scenic roads |

**Complexity:** O((V + E) log V) using a binary min-heap (`heapq`).
**Correctness:** all weights are non-negative, so Dijkstra's greedy choice is optimal.
""")
    with st.expander("🧭 Turn-by-turn directions"):
        for i, e in enumerate(res["edges"], 1):
            st.write(f"{i}. {e['text']} ({e['dist_m'] / 1000:.2f} km)")
else:
    st.info("Enter From / To locations in the sidebar, then click one of the three buttons.")
