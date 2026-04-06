# Graph Traversal Visualizer using BFS and DFS

An interactive educational visualizer built with Python + Streamlit.

## Features

- BFS and DFS traversal with step-by-step animation
- Queue/Stack state shown during each step
- Current node and visited nodes highlighted
- Graph input modes:
  - Manual input (example: `A-B, A-C, B-D`)
  - Sample graph loader
  - Random graph generator
- PyVis interactive rendering (with Matplotlib fallback)
- Comparison mode (BFS vs DFS side-by-side)
- Theory tab with complexity and use-cases

## Project Structure

- `app.py` - Streamlit app and UI
- `algorithms.py` - BFS/DFS traversal step generators
- `graph_utils.py` - graph parsing, generation, and rendering helpers
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

## Notes

- Use the sidebar to build/load a graph before running traversal.
- In Single Run mode, click **Start Traversal** to animate.
- In Comparison mode, click **Start Traversal** to run both algorithms and compare step by step.
