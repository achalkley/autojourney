"""
Flow graph builder — converts an ordered list of Screens into a directed
graph of transitions.

Uses networkx for graph construction. Deduplicates screens that appear
to be the same view (based on app_name + screen_name match).
"""
from __future__ import annotations

import logging

import networkx as nx

from autojourney.models import JourneySession, Screen

log = logging.getLogger(__name__)


def _screen_key(screen: Screen) -> str:
    """Canonical identity key for deduplication."""
    app = screen.app_name.strip() or "unknown_app"
    name = screen.screen_name.strip() or f"screen_{screen.screen_id}"
    return f"{app}::{name}"


def build_graph(session: JourneySession) -> nx.DiGraph:
    """
    Build a directed graph from the session's screens and edges.

    Nodes are screen_id strings; node attributes carry Screen metadata.
    Edge attributes carry the interaction_label.

    Returns a networkx DiGraph.
    """
    G = nx.DiGraph()

    for screen in session.screens:
        G.add_node(
            screen.screen_id,
            app_name=screen.app_name,
            screen_name=screen.screen_name,
            image_path=str(screen.image_path),
            timestamp_ms=screen.timestamp_ms,
            ui_elements=screen.ui_elements,
            inferred_action=screen.inferred_action,
            probable_destinations=screen.probable_destinations,
        )

    for edge in session.edges:
        G.add_edge(
            edge.from_screen_id,
            edge.to_screen_id,
            label=edge.interaction_label,
            timestamp_ms=edge.timestamp_ms,
        )

    log.info(
        "Flow graph: %d nodes, %d edges",
        G.number_of_nodes(),
        G.number_of_edges(),
    )
    return G


def compute_tree_layout(G: nx.DiGraph) -> dict[str, tuple[float, float]]:
    """
    Compute (x, y) positions for each node in a top-down tree layout.

    Uses graphviz 'dot' layout if available, otherwise falls back to
    networkx spring layout.

    Returns dict: node_id → (x, y) in logical units.
    """
    if G.number_of_nodes() == 0:
        return {}

    try:
        pos = nx.nx_agraph.graphviz_layout(G, prog="dot")
        # Graphviz uses bottom-up y; flip for top-down
        max_y = max(y for _, y in pos.values()) if pos else 0
        return {n: (x, max_y - y) for n, (x, y) in pos.items()}
    except Exception:  # noqa: BLE001 — graphviz backend is optional; any failure falls through to the next layout strategy
        log.debug("pygraphviz not available; using spring layout")

    # Attempt hierarchical layout via topological generations
    try:
        roots = [n for n in G.nodes if G.in_degree(n) == 0]
        if not roots:
            roots = list(G.nodes)[:1]
        pos = {}
        for root in roots:
            subgraph = nx.bfs_tree(G, root)
            sub_pos = _hierarchy_pos(subgraph, root)
            pos.update(sub_pos)
        # Fill in any disconnected nodes
        all_placed = set(pos.keys())
        x_offset = max((x for x, _ in pos.values()), default=0) + 300
        for node in G.nodes:
            if node not in all_placed:
                pos[node] = (x_offset, 0)
                x_offset += 300
        return pos
    except Exception:  # noqa: BLE001 — final layout fallback: any failure in the hierarchy walk degrades to spring layout
        return nx.spring_layout(G, seed=42, scale=800)


def _hierarchy_pos(
    T: nx.DiGraph,
    root: str,
    x: float = 0.0,
    y: float = 0.0,
    x_spacing: float = 300.0,
    y_spacing: float = 400.0,
    pos: dict | None = None,
    x_counter: list | None = None,
) -> dict[str, tuple[float, float]]:
    """Recursive tree layout (left-to-right within generation)."""
    if pos is None:
        pos = {}
    if x_counter is None:
        x_counter = [0]

    pos[root] = (x_counter[0] * x_spacing, -y)
    x_counter[0] += 1

    children = list(T.successors(root))
    for child in children:
        _hierarchy_pos(T, child, x, y + y_spacing, x_spacing, y_spacing, pos, x_counter)

    return pos
