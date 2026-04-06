import time
from typing import Dict, List

import networkx as nx
import streamlit as st
from streamlit.components.v1 import html as st_html

from algorithms import bfs_steps, dfs_steps
from graph_utils import (
    SAMPLE_EDGES,
    adjacency_dict,
    build_pyvis_html,
    draw_matplotlib,
    graph_from_edges,
    parse_edges,
    random_graph,
)


st.set_page_config(
    page_title="Graph Traversal Visualizer",
    page_icon="\U0001f9ed",
    layout="wide",
    initial_sidebar_state="expanded",
)


def inject_styles() -> None:
    st.markdown(
        """
        <style>
            .stApp {
                background:
                    radial-gradient(1100px 550px at 10% 10%, rgba(34, 211, 238, 0.20), transparent 55%),
                    radial-gradient(900px 500px at 90% 12%, rgba(56, 189, 248, 0.16), transparent 50%),
                    linear-gradient(170deg, #050816 0%, #0B132B 50%, #101D42 100%);
                color: #E2E8F0;
            }
            h1, h2, h3 {
                letter-spacing: 0.2px;
            }
            .panel {
                background: rgba(15, 23, 42, 0.75);
                border: 1px solid rgba(148, 163, 184, 0.2);
                border-radius: 14px;
                padding: 1rem 1.1rem;
                backdrop-filter: blur(4px);
            }
            .metric-chip {
                display: inline-block;
                padding: 0.35rem 0.7rem;
                margin: 0.25rem 0.25rem 0.25rem 0;
                border-radius: 999px;
                font-size: 0.9rem;
                background: rgba(30, 41, 59, 0.95);
                border: 1px solid rgba(148, 163, 184, 0.4);
            }
            .step-log {
                border-left: 3px solid #22D3EE;
                padding-left: 0.7rem;
                margin: 0.45rem 0;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _safe_pos(graph: nx.Graph, seed: int = 11) -> Dict[str, List[float]]:
    if graph.number_of_nodes() == 0:
        return {}
    return nx.spring_layout(graph, seed=seed, k=1.4 / max(1, (graph.number_of_nodes() ** 0.5)))


def initialize_state() -> None:
    defaults = {
        "graph": nx.Graph(),
        "graph_built": False,
        "steps": [],
        "current_step": -1,
        "running": False,
        "mode": "Single Run",
        "algo": "BFS",
        "start_node": "",
        "viz_engine": "PyVis",
        "render_traversed_edges": [],
        "render_active_edge": None,
        "render_step_type": "",
        "comparison": {},
        "logs": [],
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def reset_run_state() -> None:
    st.session_state.steps = []
    st.session_state.current_step = -1
    st.session_state.running = False
    st.session_state.logs = []
    st.session_state.render_traversed_edges = []
    st.session_state.render_active_edge = None
    st.session_state.render_step_type = ""


def build_graph_from_sidebar() -> None:
    st.sidebar.subheader("Graph Input")
    input_mode = st.sidebar.radio(
        "Choose graph source",
        options=["Manual Input", "Sample Graph", "Random Graph"],
        index=0,
    )

    graph = nx.Graph()

    if input_mode == "Manual Input":
        raw_edges = st.sidebar.text_area(
            "Enter edges (comma or newline separated)",
            value=SAMPLE_EDGES,
            help="Example: A-B, A-C, B-D",
            height=130,
        )
        if st.sidebar.button("Build Graph", use_container_width=True):
            try:
                edges = parse_edges(raw_edges)
                graph = graph_from_edges(edges)
                st.session_state.graph = graph
                st.session_state.graph_built = True
                reset_run_state()
                st.sidebar.success("Graph built successfully.")
            except ValueError as exc:
                st.sidebar.error(str(exc))

    elif input_mode == "Sample Graph":
        if st.sidebar.button("Load Sample Graph", use_container_width=True):
            try:
                edges = parse_edges(SAMPLE_EDGES)
                graph = graph_from_edges(edges)
                st.session_state.graph = graph
                st.session_state.graph_built = True
                reset_run_state()
                st.sidebar.success("Sample graph loaded.")
            except ValueError as exc:
                st.sidebar.error(str(exc))

        st.sidebar.caption(f"Sample edges: {SAMPLE_EDGES}")

    else:
        node_count = st.sidebar.slider("Nodes", min_value=4, max_value=16, value=8)
        edge_probability = st.sidebar.slider("Edge probability", min_value=0.10, max_value=0.65, value=0.28, step=0.01)
        random_seed = st.sidebar.number_input("Random seed", min_value=0, max_value=9999, value=7, step=1)
        if st.sidebar.button("Generate Random Graph", use_container_width=True):
            graph = random_graph(node_count=node_count, edge_probability=edge_probability, seed=int(random_seed))
            st.session_state.graph = graph
            st.session_state.graph_built = True
            reset_run_state()
            st.sidebar.success("Random graph generated.")


def build_controls_sidebar() -> None:
    st.sidebar.subheader("Traversal Controls")

    st.session_state.viz_engine = st.sidebar.radio(
        "Visualization engine",
        options=["PyVis", "Matplotlib"],
        horizontal=True,
    )

    st.session_state.mode = st.sidebar.radio(
        "Mode",
        options=["Single Run", "Comparison"],
        horizontal=False,
    )

    st.session_state.algo = st.sidebar.radio(
        "Algorithm",
        options=["BFS", "DFS"],
        horizontal=True,
        disabled=st.session_state.mode == "Comparison",
    )

    speed_label = st.sidebar.select_slider(
        "Animation speed",
        options=["Slow", "Medium", "Fast"],
        value="Medium",
    )
    speed_map = {"Slow": 1.1, "Medium": 0.55, "Fast": 0.2}
    st.session_state.speed_seconds = speed_map[speed_label]

    if st.session_state.graph.number_of_nodes() > 0:
        nodes = sorted(list(st.session_state.graph.nodes()))
        default_index = 0
        if st.session_state.start_node in nodes:
            default_index = nodes.index(st.session_state.start_node)
        st.session_state.start_node = st.sidebar.selectbox("Start node", options=nodes, index=default_index)
    else:
        st.session_state.start_node = ""
        st.sidebar.selectbox("Start node", options=["(build graph first)"], index=0, disabled=True)

    col_a, col_b = st.sidebar.columns(2)
    with col_a:
        if st.button("Start Traversal", use_container_width=True):
            start_traversal()
    with col_b:
        if st.button("Reset", use_container_width=True):
            reset_run_state()
            st.session_state.comparison = {}
            st.rerun()


def start_traversal() -> None:
    graph = st.session_state.graph
    if graph.number_of_nodes() == 0:
        st.sidebar.error("Graph is empty. Build or load a graph first.")
        return

    start = st.session_state.start_node
    if not start:
        st.sidebar.error("Select a valid start node.")
        return

    adj = adjacency_dict(graph)
    reset_run_state()

    try:
        if st.session_state.mode == "Comparison":
            bfs = bfs_steps(adj, start)
            dfs = dfs_steps(adj, start)
            st.session_state.comparison = {"BFS": bfs, "DFS": dfs}
            st.session_state.steps = []
            st.session_state.current_step = -1
            st.session_state.running = False
        else:
            if st.session_state.algo == "BFS":
                st.session_state.steps = bfs_steps(adj, start)
            else:
                st.session_state.steps = dfs_steps(adj, start)
            st.session_state.running = True
            st.session_state.current_step = -1
    except ValueError as exc:
        st.sidebar.error(str(exc))


def render_graph(graph: nx.Graph, visited: List[str], current: str) -> None:
    if graph.number_of_nodes() == 0:
        st.info("Build or load a graph from the sidebar to begin.")
        return

    visited_set = set(visited)
    pos = _safe_pos(graph)
    renderer = st.session_state.viz_engine

    if renderer == "PyVis":
        html = build_pyvis_html(
            graph=graph,
            pos=pos,
            visited=visited_set,
            current=current,
            traversed_edges=set(st.session_state.get("render_traversed_edges", [])),
            active_edge=st.session_state.get("render_active_edge"),
            active_step_type=st.session_state.get("render_step_type", ""),
        )
        st_html(html, height=580, scrolling=True)
    else:
        fig = draw_matplotlib(
            graph=graph,
            pos=pos,
            visited=visited_set,
            current=current,
            traversed_edges=set(st.session_state.get("render_traversed_edges", [])),
            active_edge=st.session_state.get("render_active_edge"),
            active_step_type=st.session_state.get("render_step_type", ""),
        )
        st.pyplot(fig, use_container_width=True)


def animate_single_run() -> None:
    steps = st.session_state.steps
    graph = st.session_state.graph
    algo = st.session_state.algo

    if not steps:
        st.warning("No traversal steps available. Click Start Traversal.")
        return

    graph_col, data_col = st.columns([2.1, 1.2], gap="large")

    graph_slot = graph_col.empty()
    status_slot = data_col.container()
    log_slot = st.container()

    for idx, step in enumerate(steps):
        st.session_state.current_step = idx
        st.session_state.logs.append(step.explanation)
        st.session_state.render_traversed_edges = step.traversed_edges
        st.session_state.render_active_edge = step.traversal_edge
        st.session_state.render_step_type = step.step_type

        with graph_slot.container():
            st.subheader(f"{algo} Traversal - Step {idx + 1}/{len(steps)}")
            render_graph(graph, step.seen_nodes, step.current)

        with status_slot:
            st.markdown('<div class="panel">', unsafe_allow_html=True)
            st.markdown(f"#### Current Node: {step.current}")
            st.markdown(
                f'<span class="metric-chip">Traversal Order: {", ".join(step.traversal_order) or "None"}</span>',
                unsafe_allow_html=True,
            )
            st.markdown(
                f'<span class="metric-chip">Seen Nodes: {", ".join(step.seen_nodes) or "None"}</span>',
                unsafe_allow_html=True,
            )
            ds_name = "Queue" if algo == "BFS" else "Stack"
            st.markdown(f"<span class=\"metric-chip\">{ds_name}: [{', '.join(step.frontier)}]</span>", unsafe_allow_html=True)
            if step.traversal_edge:
                edge_label = f"{step.traversal_edge[0]} -> {step.traversal_edge[1]}"
                st.markdown(f"<span class=\"metric-chip\">Active Edge: {edge_label}</span>", unsafe_allow_html=True)
            st.markdown(f"<span class=\"metric-chip\">Event: {step.step_type}</span>", unsafe_allow_html=True)
            st.info(step.explanation)
            st.markdown("</div>", unsafe_allow_html=True)

        with log_slot:
            st.markdown("### Step Explanations")
            recent_logs = st.session_state.logs[-8:]
            for item in recent_logs:
                st.markdown(f"<div class='step-log'>{item}</div>", unsafe_allow_html=True)

        time.sleep(st.session_state.speed_seconds)

    st.success(f"{algo} complete: {' -> '.join(steps[-1].traversal_order)}")
    st.session_state.running = False


def render_single_static_state() -> None:
    graph = st.session_state.graph
    steps = st.session_state.steps
    algo = st.session_state.algo

    if not steps:
        st.subheader("Graph Visualization")
        render_graph(graph, [], "")
        return

    step = steps[min(max(st.session_state.current_step, 0), len(steps) - 1)]
    st.session_state.render_traversed_edges = step.traversed_edges
    st.session_state.render_active_edge = step.traversal_edge
    st.session_state.render_step_type = step.step_type
    left, right = st.columns([2.1, 1.2], gap="large")

    with left:
        st.subheader(f"{algo} Traversal State")
        render_graph(graph, step.seen_nodes, step.current)

    with right:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.markdown(f"#### Current Node: {step.current}")
        st.write(f"Traversal Order: {', '.join(step.traversal_order)}")
        st.write(f"Seen Nodes: {', '.join(step.seen_nodes)}")
        ds_name = "Queue" if algo == "BFS" else "Stack"
        st.write(f"{ds_name}: [{', '.join(step.frontier)}]")
        if step.traversal_edge:
            st.write(f"Active Edge: {step.traversal_edge[0]} -> {step.traversal_edge[1]}")
        st.write(f"Event Type: {step.step_type}")
        st.info(step.explanation)
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("### Traversal Output")
    st.code(f"{algo}: {' -> '.join(steps[-1].traversal_order)}")


def render_comparison_mode() -> None:
    graph = st.session_state.graph
    result = st.session_state.comparison

    if not result:
        st.info("Click Start Traversal in Comparison mode to run BFS and DFS side-by-side.")
        render_graph(graph, [], "")
        return

    bfs_result = result["BFS"]
    dfs_result = result["DFS"]

    col1, col2 = st.columns(2, gap="large")
    max_steps = max(len(bfs_result), len(dfs_result))
    step_idx = st.slider("Comparison step", min_value=1, max_value=max_steps, value=1)

    with col1:
        st.subheader("BFS View")
        bfs_step = bfs_result[min(step_idx - 1, len(bfs_result) - 1)]
        st.session_state.render_traversed_edges = bfs_step.traversed_edges
        st.session_state.render_active_edge = bfs_step.traversal_edge
        st.session_state.render_step_type = bfs_step.step_type
        render_graph(graph, bfs_step.seen_nodes, bfs_step.current)
        st.write(f"Queue: [{', '.join(bfs_step.frontier)}]")
        st.info(bfs_step.explanation)
        st.code(f"BFS: {' -> '.join(bfs_result[-1].traversal_order)}")

    with col2:
        st.subheader("DFS View")
        dfs_step = dfs_result[min(step_idx - 1, len(dfs_result) - 1)]
        st.session_state.render_traversed_edges = dfs_step.traversed_edges
        st.session_state.render_active_edge = dfs_step.traversal_edge
        st.session_state.render_step_type = dfs_step.step_type
        render_graph(graph, dfs_step.seen_nodes, dfs_step.current)
        st.write(f"Stack: [{', '.join(dfs_step.frontier)}]")
        st.info(dfs_step.explanation)
        st.code(f"DFS: {' -> '.join(dfs_result[-1].traversal_order)}")


def render_theory_tab() -> None:
    st.subheader("Breadth-First Search (BFS)")
    st.markdown(
        """
        <div class="panel">
        BFS explores the graph level by level from the start node.<br>
        It uses a <b>Queue (FIFO)</b>, so the earliest discovered node is processed first.
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.subheader("Depth-First Search (DFS)")
    st.markdown(
        """
        <div class="panel">
        DFS explores as deep as possible before backtracking.<br>
        It uses a <b>Stack (LIFO)</b> in iterative implementations.
        </div>
        """,
        unsafe_allow_html=True,
    )

    left, right = st.columns(2, gap="large")
    with left:
        st.markdown("### Key Differences")
        st.markdown(
            """
            - BFS is ideal for shortest-path in unweighted graphs.
            - DFS is ideal for deep exploration, cycle detection, and topological ideas.
            - BFS memory can grow large on wide graphs.
            - DFS can be memory efficient on sparse deep graphs.
            """
        )
    with right:
        st.markdown("### Complexity")
        st.markdown(
            """
            - Time Complexity (both): **O(V + E)**
            - Space Complexity BFS: **O(V)** for queue/visited
            - Space Complexity DFS: **O(V)** for stack/visited
            """
        )

    st.markdown("### Real-World Use Cases")
    st.markdown(
        """
        - BFS: social network degree search, shortest path in maps with equal weights.
        - DFS: dependency analysis, puzzle solving, maze traversal, cycle detection.
        """
    )


def main() -> None:
    inject_styles()
    initialize_state()

    st.title("Graph Traversal Visualizer: BFS and DFS")
    st.caption("Interactive educational tool built with Python, Streamlit, NetworkX, and PyVis.")

    build_graph_from_sidebar()
    build_controls_sidebar()

    tab_visualizer, tab_theory = st.tabs(["Visualizer", "Theory"])

    with tab_visualizer:
        if st.session_state.mode == "Comparison":
            render_comparison_mode()
        else:
            if st.session_state.running:
                animate_single_run()
            else:
                render_single_static_state()

        st.divider()
        st.subheader("Adjacency List")
        if st.session_state.graph.number_of_nodes() > 0:
            adj = adjacency_dict(st.session_state.graph)
            st.json(adj)
        else:
            st.info("Adjacency list will appear here once a graph is loaded.")

    with tab_theory:
        render_theory_tab()


if __name__ == "__main__":
    main()
