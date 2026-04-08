import re
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Dict, List, Set, Tuple

import matplotlib.pyplot as plt
import networkx as nx
from pyvis.network import Network


SAMPLE_EDGES = "A-B, A-C, B-D, B-E, C-F, E-G, F-H"
BEST_FIRST_SAMPLE_EDGES = "A-B, A-C, B-D, B-E, C-F, C-G, F-H, G-I"
BEST_FIRST_SAMPLE_HEURISTICS = {
    "A": 7,
    "B": 6,
    "C": 4,
    "D": 8,
    "E": 5,
    "F": 2,
    "G": 1,
    "H": 3,
    "I": 0,
}


def parse_edges(text: str) -> List[Tuple[str, str]]:
    """Parse comma/newline-separated edge tokens like A-B."""
    if not text or not text.strip():
        return []

    cleaned = text.replace("\n", ",")
    tokens = [token.strip() for token in cleaned.split(",") if token.strip()]

    edges: List[Tuple[str, str]] = []
    pattern = re.compile(r"^([A-Za-z0-9_]+)\s*[-:]\s*([A-Za-z0-9_]+)$")

    for token in tokens:
        match = pattern.match(token)
        if not match:
            raise ValueError(
                f"Invalid edge token '{token}'. Use format like A-B, B-C, C-D."
            )

        src, dst = match.group(1), match.group(2)
        if src == dst:
            raise ValueError(f"Self-loop '{src}-{dst}' is not allowed in this demo.")

        edges.append((src, dst))

    if not edges:
        raise ValueError("No valid edges found.")

    return edges


def graph_from_edges(edges: List[Tuple[str, str]]) -> nx.Graph:
    graph = nx.Graph()
    graph.add_edges_from(edges)
    return graph


def adjacency_dict(graph: nx.Graph) -> Dict[str, List[str]]:
    return {node: sorted(list(graph.neighbors(node))) for node in sorted(graph.nodes())}


def random_graph(node_count: int = 8, edge_probability: float = 0.28, seed: int = 7) -> nx.Graph:
    graph = nx.erdos_renyi_graph(node_count, edge_probability, seed=seed)

    # Ensure node labels are friendly strings.
    relabel_map = {node: chr(65 + node) for node in graph.nodes()}
    graph = nx.relabel_nodes(graph, relabel_map)

    # Guarantee at least one edge for visualization usefulness.
    if graph.number_of_edges() == 0 and graph.number_of_nodes() >= 2:
        nodes = sorted(graph.nodes())
        graph.add_edge(nodes[0], nodes[1])

    return graph


def _node_styles(node: str, visited: Set[str], current: str) -> Tuple[str, int]:
    if node == current:
        return "#F97316", 34
    if node in visited:
        return "#10B981", 28
    return "#CBD5E1", 24


def _edge_key(a: str, b: str) -> Tuple[str, str]:
    return tuple(sorted((a, b)))


def build_pyvis_html(
    graph: nx.Graph,
    pos: Dict[str, Tuple[float, float]],
    visited: Set[str],
    current: str,
    traversed_edges: Set[Tuple[str, str]],
    active_edge: Tuple[str, str] | None,
    active_step_type: str = "",
    heuristics: Dict[str, int] | None = None,
    final_path_edges: Set[Tuple[str, str]] | None = None,
) -> str:
    net = Network(height="560px", width="100%", directed=False, bgcolor="#0F172A", font_color="#E2E8F0")
    net.barnes_hut()

    for node in sorted(graph.nodes()):
        color, size = _node_styles(node, visited, current)
        x, y = pos[node]
        node_label = f"{node}({heuristics[node]})" if heuristics and node in heuristics else node
        title = f"Node: {node}"
        if heuristics and node in heuristics:
            title = f"Node: {node} | h={heuristics[node]}"

        net.add_node(
            node,
            label=node_label,
            title=title,
            color=color,
            size=size,
            x=float(x) * 340,
            y=float(y) * 340,
            physics=False,
        )

    for source, target in graph.edges():
        edge = _edge_key(source, target)
        color = "#94A3B8"
        width = 2
        dashes = False

        if edge in traversed_edges:
            color = "#22C55E"
            width = 3

        if final_path_edges and edge in final_path_edges:
            color = "#EC4899"
            width = 5

        if active_edge and edge == active_edge:
            if active_step_type == "backtrack":
                color = "#F59E0B"
                dashes = True
            else:
                color = "#38BDF8"
            width = 5

        net.add_edge(source, target, color=color, width=width, dashes=dashes)

    net.set_options(
        """
        {
          "nodes": {
            "shape": "dot",
            "font": {
              "size": 20,
              "face": "Trebuchet MS",
              "color": "#F8FAFC"
            },
            "borderWidth": 2,
            "borderWidthSelected": 3
          },
          "edges": {
            "smooth": {
              "type": "continuous"
            },
            "selectionWidth": 3
          },
          "interaction": {
            "hover": true,
            "navigationButtons": true,
            "keyboard": true
          },
          "physics": {
            "enabled": false
          }
        }
        """
    )

    with NamedTemporaryFile(delete=False, suffix=".html") as temp_file:
        temp_path = Path(temp_file.name)

    net.save_graph(str(temp_path))
    html = temp_path.read_text(encoding="utf-8")
    temp_path.unlink(missing_ok=True)
    return html


def draw_matplotlib(
    graph: nx.Graph,
    pos: Dict[str, Tuple[float, float]],
    visited: Set[str],
    current: str,
    traversed_edges: Set[Tuple[str, str]],
    active_edge: Tuple[str, str] | None,
    active_step_type: str = "",
    heuristics: Dict[str, int] | None = None,
    final_path_edges: Set[Tuple[str, str]] | None = None,
):
    node_colors = []
    node_sizes = []
    for node in sorted(graph.nodes()):
        color, size = _node_styles(node, visited, current)
        node_colors.append(color)
        node_sizes.append(size * 34)

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.set_facecolor("#0B1020")
    fig.patch.set_facecolor("#0B1020")

    nx.draw_networkx_edges(graph, pos, edge_color="#94A3B8", width=2.0, ax=ax)

    traversed = [
        edge
        for edge in graph.edges()
        if _edge_key(edge[0], edge[1]) in traversed_edges
    ]
    if traversed:
        nx.draw_networkx_edges(graph, pos, edgelist=traversed, edge_color="#22C55E", width=3.0, ax=ax)

    if final_path_edges:
        final_path = [
            edge
            for edge in graph.edges()
            if _edge_key(edge[0], edge[1]) in final_path_edges
        ]
        if final_path:
            nx.draw_networkx_edges(graph, pos, edgelist=final_path, edge_color="#EC4899", width=5.2, ax=ax)

    if active_edge:
        active = [
            edge
            for edge in graph.edges()
            if _edge_key(edge[0], edge[1]) == active_edge
        ]
        if active:
            style = "dashed" if active_step_type == "backtrack" else "solid"
            color = "#F59E0B" if active_step_type == "backtrack" else "#38BDF8"
            nx.draw_networkx_edges(
                graph,
                pos,
                edgelist=active,
                edge_color=color,
                width=4.5,
                style=style,
                ax=ax,
            )
    nx.draw_networkx_nodes(
        graph,
        pos,
        nodelist=sorted(graph.nodes()),
        node_color=node_colors,
        node_size=node_sizes,
        linewidths=2,
        edgecolors="#F8FAFC",
        ax=ax,
    )
    nx.draw_networkx_labels(
        graph,
        pos,
        labels={
            node: (f"{node}({heuristics[node]})" if heuristics and node in heuristics else node)
            for node in graph.nodes()
        },
        font_size=14,
        font_color="#F8FAFC",
        font_weight="bold",
        ax=ax,
    )

    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    plt.tight_layout()
    return fig
