# Graph Traversal Visualizer using BFS, DFS, and Greedy Best-First Search

An interactive educational visualizer built with Python + Streamlit.

## Features

- BFS traversal with queue visualization
- DFS traversal with stack and backtracking visualization
- Greedy Best-First Search traversal with:
  - heuristic input (manual)
  - auto heuristics by distance-to-goal
  - sample demo heuristics
  - open list (priority queue) display
  - "why selected" explanation per step
  - goal-based stopping and final path reconstruction
- Node and edge highlighting:
  - current node
  - active traversal edge
  - explored/traversed edges
  - final successful path (goal mode)
- Graph input modes:
  - Manual input (example: `A-B, A-C, B-D`)
  - Sample graph loader
  - Best-First demo graph loader
  - Random graph generator
- PyVis interactive rendering (with Matplotlib fallback)
- Comparison mode (BFS vs DFS side-by-side)
- Theory tab with complexity and use-cases

## Project Structure

- `app.py` - Streamlit app and UI
- `algorithms.py` - BFS/DFS/Best-First traversal step generators
- `graph_utils.py` - graph parsing, generation, and rendering helpers/constants
- `requirements.txt` - dependencies

## Run Locally

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Start Streamlit app:

```bash
streamlit run app.py
```

## Sample Manual Input

```text
A-B, A-C, B-D, B-E, C-F, E-G, F-H
```

## Best-First Demo Data

Graph edges:

```text
A-B, A-C, B-D, B-E, C-F, C-G, F-H, G-I
```

Heuristics:

```text
A=7
B=6
C=4
D=8
E=5
F=2
G=1
H=3
I=0
```

## How To Test Best-First Search

1. Run the app and choose **Best-First Demo Graph** in the sidebar.
2. Set **Algorithm** to **Greedy Best-First Search**.
3. Select **Start node** and **Goal node** (example: start `A`, goal `I`).
4. Choose heuristic mode:
  - `Sample Demo Heuristics` to use the preset values above
  - `Auto by Goal Distance` for generated heuristics
  - `Manual Input` to type your own values
5. Click **Start Traversal**.
6. Observe each step:
  - current selected node
  - open list ordering by heuristic
  - active edge and explored edges
  - explanation of why the selected node was chosen
7. If goal is reached, review **Final Path**, **Total nodes visited**, and **Path length**.

## Notes

- Use the sidebar to build/load a graph before running traversal.
- In Single Run mode, click **Start Traversal** to animate.
- In Comparison mode, click **Start Traversal** to run BFS and DFS side-by-side.
- For Best-First Search, ensure every graph node has a heuristic value.
