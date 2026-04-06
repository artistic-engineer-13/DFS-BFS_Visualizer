from collections import deque
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Set, Tuple


@dataclass
class TraversalStep:
    """Represents one atomic traversal event for visualization."""

    step_type: str
    current: str
    previous: Optional[str]
    traversal_edge: Optional[Tuple[str, str]]
    traversal_order: List[str]
    seen_nodes: List[str]
    frontier: List[str]
    traversed_edges: List[Tuple[str, str]]
    explanation: str


def _edge_key(a: str, b: str) -> Tuple[str, str]:
    """Normalize undirected edge identity for consistent highlighting."""
    return tuple(sorted((a, b)))


def bfs_steps(adjacency: Dict[str, Iterable[str]], start: str) -> List[TraversalStep]:
    """Iterative BFS with detailed visualization events."""
    if start not in adjacency:
        raise ValueError(f"Start node '{start}' is not present in the graph.")

    visited: Set[str] = {start}
    seen_nodes: List[str] = [start]
    order: List[str] = []
    parent: Dict[str, Optional[str]] = {start: None}
    queue: deque[str] = deque([start])
    steps: List[TraversalStep] = []
    traversed_edges: List[Tuple[str, str]] = []

    steps.append(
        TraversalStep(
            step_type="start",
            current=start,
            previous=None,
            traversal_edge=None,
            traversal_order=order.copy(),
            seen_nodes=seen_nodes.copy(),
            frontier=list(queue),
            traversed_edges=traversed_edges.copy(),
            explanation=f"Started BFS from node {start}. Queue initialized with [{start}].",
        )
    )

    while queue:
        current = queue.popleft()
        order.append(current)

        parent_node = parent[current]
        active_edge = _edge_key(parent_node, current) if parent_node else None
        steps.append(
            TraversalStep(
                step_type="visit",
                current=current,
                previous=parent_node,
                traversal_edge=active_edge,
                traversal_order=order.copy(),
                seen_nodes=seen_nodes.copy(),
                frontier=list(queue),
                traversed_edges=traversed_edges.copy(),
                explanation=(
                    f"Dequeued and visited {current}. "
                    f"Now exploring neighbors of {current} in sorted order."
                ),
            )
        )

        newly_added: List[str] = []
        for neighbor in sorted(adjacency.get(current, [])):
            if neighbor not in visited:
                visited.add(neighbor)
                seen_nodes.append(neighbor)
                parent[neighbor] = current
                queue.append(neighbor)
                newly_added.append(neighbor)
                edge = _edge_key(current, neighbor)
                traversed_edges.append(edge)

                steps.append(
                    TraversalStep(
                        step_type="discover",
                        current=neighbor,
                        previous=current,
                        traversal_edge=edge,
                        traversal_order=order.copy(),
                        seen_nodes=seen_nodes.copy(),
                        frontier=list(queue),
                        traversed_edges=traversed_edges.copy(),
                        explanation=(
                            f"Discovered {neighbor} from {current}; "
                            f"enqueued {neighbor} for a future BFS level."
                        ),
                    )
                )

        if newly_added:
            msg = f"Visited node {current}, added neighbors {', '.join(newly_added)} to queue."
        else:
            msg = f"Visited node {current}, no new neighbors were added to queue."

        steps.append(
            TraversalStep(
                step_type="post_visit",
                current=current,
                previous=parent_node,
                traversal_edge=active_edge,
                traversal_order=order.copy(),
                seen_nodes=seen_nodes.copy(),
                frontier=list(queue),
                traversed_edges=traversed_edges.copy(),
                explanation=msg,
            )
        )

    return steps


def dfs_steps(adjacency: Dict[str, Iterable[str]], start: str) -> List[TraversalStep]:
    """Iterative DFS with movement and optional backtracking events."""
    if start not in adjacency:
        raise ValueError(f"Start node '{start}' is not present in the graph.")

    visited: Set[str] = {start}
    seen_nodes: List[str] = [start]
    order: List[str] = [start]
    stack: List[Tuple[str, Iterable[str]]] = [
        (start, iter(sorted(adjacency.get(start, []))))
    ]
    steps: List[TraversalStep] = []
    traversed_edges: List[Tuple[str, str]] = []

    steps.append(
        TraversalStep(
            step_type="start",
            current=start,
            previous=None,
            traversal_edge=None,
            traversal_order=order.copy(),
            seen_nodes=seen_nodes.copy(),
            frontier=[node for node, _ in stack],
            traversed_edges=traversed_edges.copy(),
            explanation=f"Started DFS at node {start}. Stack initialized with [{start}].",
        )
    )

    while stack:
        current, neighbors_iter = stack[-1]

        try:
            neighbor = next(neighbors_iter)
        except StopIteration:
            completed = stack.pop()[0]
            if stack:
                parent = stack[-1][0]
                backtrack_edge = _edge_key(completed, parent)
                steps.append(
                    TraversalStep(
                        step_type="backtrack",
                        current=parent,
                        previous=completed,
                        traversal_edge=backtrack_edge,
                        traversal_order=order.copy(),
                        seen_nodes=seen_nodes.copy(),
                        frontier=[node for node, _ in stack],
                        traversed_edges=traversed_edges.copy(),
                        explanation=(
                            f"Backtracked from {completed} to {parent} "
                            f"after exploring all neighbors of {completed}."
                        ),
                    )
                )
            continue

        edge = _edge_key(current, neighbor)
        if neighbor in visited:
            steps.append(
                TraversalStep(
                    step_type="skip",
                    current=current,
                    previous=neighbor,
                    traversal_edge=edge,
                    traversal_order=order.copy(),
                    seen_nodes=seen_nodes.copy(),
                    frontier=[node for node, _ in stack],
                    traversed_edges=traversed_edges.copy(),
                    explanation=(
                        f"Skipped edge {current}-{neighbor} because {neighbor} "
                        "was already visited."
                    ),
                )
            )
            continue

        visited.add(neighbor)
        seen_nodes.append(neighbor)
        order.append(neighbor)
        traversed_edges.append(edge)
        stack.append((neighbor, iter(sorted(adjacency.get(neighbor, [])))))

        steps.append(
            TraversalStep(
                step_type="discover",
                current=neighbor,
                previous=current,
                traversal_edge=edge,
                traversal_order=order.copy(),
                seen_nodes=seen_nodes.copy(),
                frontier=[node for node, _ in stack],
                traversed_edges=traversed_edges.copy(),
                explanation=(
                    f"Moved from {current} to {neighbor} and pushed {neighbor} onto stack."
                ),
            )
        )

    return steps
