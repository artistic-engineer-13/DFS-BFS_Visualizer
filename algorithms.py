from collections import deque
from dataclasses import dataclass
import heapq
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
    selection_reason: str = ""
    goal: Optional[str] = None
    goal_found: bool = False
    final_path: List[str] = None
    final_path_edges: List[Tuple[str, str]] = None

    def __post_init__(self) -> None:
        if self.final_path is None:
            self.final_path = []
        if self.final_path_edges is None:
            self.final_path_edges = []


def _edge_key(a: str, b: str) -> Tuple[str, str]:
    """Normalize undirected edge identity for consistent highlighting."""
    return tuple(sorted((a, b)))


def _format_heap_frontier(heap: List[Tuple[int, str]], heuristics: Dict[str, int]) -> List[str]:
    ordered = sorted(heap, key=lambda item: (item[0], item[1]))
    return [f"{node}({heuristics[node]})" for _, node in ordered]


def _reconstruct_path(parent: Dict[str, Optional[str]], target: str) -> List[str]:
    path = [target]
    node = target
    while parent.get(node) is not None:
        node = parent[node]  # type: ignore[index]
        path.append(node)
    path.reverse()
    return path


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


def best_first_steps(
    adjacency: Dict[str, Iterable[str]],
    start: str,
    heuristics: Dict[str, int],
    goal: Optional[str] = None,
    stop_on_goal: bool = True,
) -> List[TraversalStep]:
    """Greedy Best-First Search using a min-heap prioritized by heuristic values."""
    if start not in adjacency:
        raise ValueError(f"Start node '{start}' is not present in the graph.")

    missing = [node for node in adjacency if node not in heuristics]
    if missing:
        raise ValueError(f"Missing heuristic values for nodes: {', '.join(sorted(missing))}")

    if goal and goal not in adjacency:
        raise ValueError(f"Goal node '{goal}' is not present in the graph.")

    visited: Set[str] = set()
    seen_nodes: List[str] = [start]
    order: List[str] = []
    parent: Dict[str, Optional[str]] = {start: None}
    traversed_edges: List[Tuple[str, str]] = []
    steps: List[TraversalStep] = []

    frontier_heap: List[Tuple[int, str]] = [(heuristics[start], start)]
    in_frontier: Set[str] = {start}

    steps.append(
        TraversalStep(
            step_type="start",
            current=start,
            previous=None,
            traversal_edge=None,
            traversal_order=order.copy(),
            seen_nodes=seen_nodes.copy(),
            frontier=_format_heap_frontier(frontier_heap, heuristics),
            traversed_edges=traversed_edges.copy(),
            explanation=(
                f"Started Greedy Best-First Search from {start}. "
                f"Open list initialized with {start}({heuristics[start]})."
            ),
            selection_reason=(
                f"Node {start} is the only frontier node, so it is selected first."
            ),
            goal=goal,
        )
    )

    goal_found = False
    goal_path: List[str] = []
    goal_path_edges: List[Tuple[str, str]] = []

    while frontier_heap:
        score, current = heapq.heappop(frontier_heap)
        in_frontier.discard(current)

        if current in visited:
            steps.append(
                TraversalStep(
                    step_type="skip",
                    current=current,
                    previous=parent.get(current),
                    traversal_edge=None,
                    traversal_order=order.copy(),
                    seen_nodes=seen_nodes.copy(),
                    frontier=_format_heap_frontier(frontier_heap, heuristics),
                    traversed_edges=traversed_edges.copy(),
                    explanation=f"Skipped {current} because it was already visited.",
                    selection_reason=f"{current} remained in the heap but had already been processed.",
                    goal=goal,
                )
            )
            continue

        visited.add(current)
        order.append(current)
        parent_node = parent.get(current)
        active_edge = _edge_key(parent_node, current) if parent_node else None

        frontier_now = _format_heap_frontier(frontier_heap, heuristics)
        reason = (
            f"Selected {current} because h({current})={score} is the lowest in the open list"
            f" {frontier_now if frontier_now else '[]'}"
        )
        steps.append(
            TraversalStep(
                step_type="visit",
                current=current,
                previous=parent_node,
                traversal_edge=active_edge,
                traversal_order=order.copy(),
                seen_nodes=seen_nodes.copy(),
                frontier=frontier_now,
                traversed_edges=traversed_edges.copy(),
                explanation=f"Visiting node {current} with heuristic value {score}.",
                selection_reason=reason + ".",
                goal=goal,
            )
        )

        if goal and current == goal:
            goal_found = True
            goal_path = _reconstruct_path(parent, goal)
            goal_path_edges = [_edge_key(goal_path[i], goal_path[i + 1]) for i in range(len(goal_path) - 1)]

            steps.append(
                TraversalStep(
                    step_type="goal",
                    current=current,
                    previous=parent_node,
                    traversal_edge=active_edge,
                    traversal_order=order.copy(),
                    seen_nodes=seen_nodes.copy(),
                    frontier=frontier_now,
                    traversed_edges=traversed_edges.copy(),
                    explanation=(
                        f"Goal node {goal} reached. Reconstructed path: {' -> '.join(goal_path)}."
                    ),
                    selection_reason=(
                        f"Search stops because stop-on-goal is enabled and {goal} was selected."
                    ),
                    goal=goal,
                    goal_found=True,
                    final_path=goal_path.copy(),
                    final_path_edges=goal_path_edges.copy(),
                )
            )

            if stop_on_goal:
                break

        newly_added: List[str] = []
        for neighbor in sorted(adjacency.get(current, [])):
            if neighbor in visited or neighbor in in_frontier:
                continue

            parent[neighbor] = current
            in_frontier.add(neighbor)
            heapq.heappush(frontier_heap, (heuristics[neighbor], neighbor))
            seen_nodes.append(neighbor)
            edge = _edge_key(current, neighbor)
            traversed_edges.append(edge)
            newly_added.append(f"{neighbor}({heuristics[neighbor]})")

            steps.append(
                TraversalStep(
                    step_type="discover",
                    current=neighbor,
                    previous=current,
                    traversal_edge=edge,
                    traversal_order=order.copy(),
                    seen_nodes=seen_nodes.copy(),
                    frontier=_format_heap_frontier(frontier_heap, heuristics),
                    traversed_edges=traversed_edges.copy(),
                    explanation=(
                        f"Expanded {current} and added {neighbor}({heuristics[neighbor]}) to the open list."
                    ),
                    selection_reason=(
                        f"{neighbor} was not visited, so it is eligible for future selection by minimum heuristic."
                    ),
                    goal=goal,
                )
            )

        if newly_added:
            msg = f"After expanding {current}, open list is [{', '.join(_format_heap_frontier(frontier_heap, heuristics))}]."
        else:
            msg = f"Expanded {current}, but no new neighbors were added to the open list."

        steps.append(
            TraversalStep(
                step_type="post_visit",
                current=current,
                previous=parent_node,
                traversal_edge=active_edge,
                traversal_order=order.copy(),
                seen_nodes=seen_nodes.copy(),
                frontier=_format_heap_frontier(frontier_heap, heuristics),
                traversed_edges=traversed_edges.copy(),
                explanation=msg,
                selection_reason=(
                    f"Next selection will again choose the frontier node with smallest heuristic value."
                ),
                goal=goal,
                goal_found=goal_found,
                final_path=goal_path.copy(),
                final_path_edges=goal_path_edges.copy(),
            )
        )

    if goal and not goal_found:
        steps.append(
            TraversalStep(
                step_type="complete",
                current=order[-1] if order else start,
                previous=None,
                traversal_edge=None,
                traversal_order=order.copy(),
                seen_nodes=seen_nodes.copy(),
                frontier=_format_heap_frontier(frontier_heap, heuristics),
                traversed_edges=traversed_edges.copy(),
                explanation=(
                    f"Search ended without reaching goal {goal}."
                ),
                selection_reason="Open list is empty, so there are no more nodes to explore.",
                goal=goal,
                goal_found=False,
            )
        )

    return steps
