"""Tests for cascade module."""

import pytest
from src.cascade import (
    build_cascade_graph,
    propagate_fraud_signal,
    detect_cascade_event,
    trace_cascade_path,
    build_cascade_path_with_weights,
    compute_systemic_risk,
    simulate_cascade,
    run_cascade_analysis,
    CASCADE_WEIGHTS,
)


class TestCascadeGraph:
    """Tests for cascade graph construction."""

    def test_build_graph_default(self):
        """Test building graph with default weights."""
        graph = build_cascade_graph()
        assert len(graph) > 0
        assert "hope_florida" in graph

    def test_build_graph_custom(self):
        """Test building graph with custom weights."""
        weights = {
            ("a", "b"): 0.5,
            ("b", "c"): 0.3,
        }
        graph = build_cascade_graph(weights)
        assert "a" in graph
        assert "b" in graph["a"]
        assert graph["a"]["b"] == 0.5

    def test_build_graph_empty(self):
        """Test building graph with empty weights."""
        graph = build_cascade_graph({})
        assert graph == {}


class TestSignalPropagation:
    """Tests for fraud signal propagation."""

    def test_propagate_basic(self, cascade_graph):
        """Test basic signal propagation."""
        signals = propagate_fraud_signal(cascade_graph, "hope_florida", 0.8)
        assert "hope_florida" in signals
        assert signals["hope_florida"] == 0.8

    def test_propagate_spreads(self, cascade_graph):
        """Test signal spreads to connected domains."""
        signals = propagate_fraud_signal(cascade_graph, "hope_florida", 0.8)
        # Should spread to connected domains
        assert len(signals) > 1

    def test_propagate_decays(self, cascade_graph):
        """Test signal decays as it propagates."""
        signals = propagate_fraud_signal(cascade_graph, "hope_florida", 1.0)
        # Source should have highest signal
        assert signals["hope_florida"] >= max(
            v for k, v in signals.items() if k != "hope_florida"
        )

    def test_propagate_threshold(self, cascade_graph):
        """Test very weak signals don't propagate."""
        signals = propagate_fraud_signal(cascade_graph, "hope_florida", 0.01)
        # Weak signal should not spread much
        assert len(signals) <= 2


class TestCascadeDetection:
    """Tests for cascade event detection."""

    def test_detect_cascade_high_signal(self, cascade_graph):
        """Test cascade detection with high signals."""
        signals = {"hope_florida": 0.8, "corporate": 0.5, "insurance": 0.3}
        events = detect_cascade_event(signals, threshold=0.3)
        assert len(events) > 0

    def test_detect_cascade_low_signal(self):
        """Test no cascade with low signals."""
        signals = {"hope_florida": 0.1, "corporate": 0.1, "insurance": 0.1}
        events = detect_cascade_event(signals, threshold=0.3)
        assert len(events) == 0

    def test_cascade_event_attributes(self, cascade_graph):
        """Test cascade event has required attributes."""
        signals = {"hope_florida": 0.8, "corporate": 0.5, "insurance": 0.4}
        events = detect_cascade_event(signals, threshold=0.3)
        if events:
            event = events[0]
            assert event.source_domain is not None
            assert isinstance(event.affected_domains, list)
            assert event.total_propagation > 0


class TestCascadePath:
    """Tests for cascade path tracing."""

    def test_trace_path_direct(self, cascade_graph):
        """Test path tracing for directly connected nodes."""
        path = trace_cascade_path(cascade_graph, "hope_florida", "corporate")
        assert len(path) >= 2

    def test_trace_path_with_weights(self, cascade_graph):
        """Test path with weight annotations."""
        weighted_path = build_cascade_path_with_weights(
            cascade_graph, "hope_florida", "corporate"
        )
        if weighted_path:
            assert all("from" in p and "to" in p and "weight" in p for p in weighted_path)

    def test_trace_path_no_connection(self, cascade_graph):
        """Test path tracing for unconnected nodes."""
        path = trace_cascade_path(cascade_graph, "hope_florida", "nonexistent")
        assert path == []


class TestSystemicRisk:
    """Tests for systemic risk computation."""

    def test_compute_risk_empty(self):
        """Test risk with no events."""
        risk = compute_systemic_risk([])
        assert risk == 0.0

    def test_compute_risk_single(self, cascade_graph):
        """Test risk with single event."""
        from src.cascade import CascadeEvent

        event = CascadeEvent(
            source_domain="hope_florida",
            affected_domains=["corporate", "insurance"],
            total_propagation=1.5,
        )
        risk = compute_systemic_risk([event])
        assert 0.0 <= risk <= 1.0


class TestCascadeSimulation:
    """Tests for full cascade simulation."""

    def test_simulate_cascade(self, cascade_graph):
        """Test complete cascade simulation."""
        result = simulate_cascade(cascade_graph, "hope_florida", 0.8)
        assert "signals" in result
        assert "events" in result
        assert "systemic_risk" in result

    def test_run_cascade_analysis(self, seed):
        """Test cascade analysis run."""
        result = run_cascade_analysis(seed=seed)
        assert "cascade_result" in result
        assert "receipt" in result
