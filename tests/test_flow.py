"""
Tests for flow graph builder.
"""
from __future__ import annotations

from pathlib import Path

from autojourney.flow.graph import build_graph, compute_tree_layout
from autojourney.models import EventType, FlowEdge, JourneySession, Screen


def _make_session() -> JourneySession:
    screens = [
        Screen("s0001", Path("a.png"), 0, EventType.TRANSITION, app_name="MyApp", screen_name="Home"),
        Screen("s0002", Path("b.png"), 1000, EventType.TRANSITION, app_name="MyApp", screen_name="Detail"),
        Screen("s0003", Path("c.png"), 2000, EventType.TRANSITION, app_name="MyApp", screen_name="Checkout"),
    ]
    edges = [
        FlowEdge("s0001", "s0002", "Tapped item", 1000),
        FlowEdge("s0002", "s0003", "Tapped 'Buy'", 2000),
    ]
    return JourneySession("test_session", Path("video.mp4"), screens=screens, edges=edges)


class TestBuildGraph:
    def test_correct_node_count(self):
        session = _make_session()
        G = build_graph(session)
        assert G.number_of_nodes() == 3

    def test_correct_edge_count(self):
        session = _make_session()
        G = build_graph(session)
        assert G.number_of_edges() == 2

    def test_edge_labels(self):
        session = _make_session()
        G = build_graph(session)
        assert G["s0001"]["s0002"]["label"] == "Tapped item"

    def test_node_attributes(self):
        session = _make_session()
        G = build_graph(session)
        assert G.nodes["s0001"]["screen_name"] == "Home"
        assert G.nodes["s0001"]["app_name"] == "MyApp"


class TestComputeTreeLayout:
    def test_all_nodes_have_positions(self):
        session = _make_session()
        G = build_graph(session)
        layout = compute_tree_layout(G)
        for node in G.nodes:
            assert node in layout
            x, y = layout[node]
            assert isinstance(x, (int, float))
            assert isinstance(y, (int, float))

    def test_empty_graph_returns_empty(self):
        import networkx as nx

        from autojourney.flow.graph import compute_tree_layout
        layout = compute_tree_layout(nx.DiGraph())
        assert layout == {}
