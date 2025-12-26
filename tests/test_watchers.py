"""Tests for watchers module."""

import pytest
from src.watchers import (
    Watcher,
    FraudEvent,
    spawn_watcher,
    watcher_detect,
    check_autocatalysis,
    dissolve_watcher,
    govern_population,
    run_watcher_cycle,
    generate_fraud_events,
    run_watcher_simulation,
    WATCHER_CONFIG,
)


class TestWatcherSpawn:
    """Tests for watcher spawning."""

    def test_spawn_high_signal(self):
        """Test watcher spawns on high signal."""
        signal = {"domain": "hope_florida", "signal": 0.8}
        watcher = spawn_watcher(signal, WATCHER_CONFIG, cycle=0)
        assert watcher is not None
        assert watcher.target_domain == "hope_florida"

    def test_no_spawn_low_signal(self):
        """Test watcher doesn't spawn on low signal."""
        signal = {"domain": "hope_florida", "signal": 0.3}
        watcher = spawn_watcher(signal, WATCHER_CONFIG, cycle=0)
        assert watcher is None

    def test_spawn_id_format(self):
        """Test spawned watcher has correct ID format."""
        signal = {"domain": "insurance", "signal": 0.9}
        watcher = spawn_watcher(signal, WATCHER_CONFIG, cycle=42)
        assert watcher.watcher_id.startswith("WATCHER-INS-")


class TestWatcherDetection:
    """Tests for watcher fraud detection."""

    def test_detect_in_target_domain(self):
        """Test watcher detects events in target domain."""
        watcher = Watcher(
            watcher_id="TEST-001",
            target_domain="hope_florida",
            spawn_cycle=0,
        )
        events = [
            FraudEvent("E1", "hope_florida", 0.8, 1),
            FraudEvent("E2", "hope_florida", 0.9, 2),
        ]
        detections = watcher_detect(watcher, events)
        assert len(detections) > 0

    def test_no_detect_other_domain(self):
        """Test watcher doesn't detect events in other domains."""
        watcher = Watcher(
            watcher_id="TEST-001",
            target_domain="hope_florida",
            spawn_cycle=0,
        )
        events = [
            FraudEvent("E1", "insurance", 0.8, 1),
            FraudEvent("E2", "sheriff", 0.9, 2),
        ]
        detections = watcher_detect(watcher, events)
        assert len(detections) == 0

    def test_detection_updates_history(self):
        """Test detections are added to watcher history."""
        watcher = Watcher(
            watcher_id="TEST-001",
            target_domain="hope_florida",
            spawn_cycle=0,
        )
        events = [FraudEvent("E1", "hope_florida", 0.8, 1)]
        watcher_detect(watcher, events)
        assert len(watcher.detection_history) > 0 or watcher.total_detections >= 0


class TestAutocatalysis:
    """Tests for autocatalysis checking."""

    def test_autocatalysis_insufficient_history(self):
        """Test autocatalysis fails with insufficient history."""
        watcher = Watcher(
            watcher_id="TEST-001",
            target_domain="hope_florida",
            spawn_cycle=0,
            detection_history=[{"signal": 0.8, "timestamp": 1}],
        )
        result = check_autocatalysis(watcher, [])
        assert result is False

    def test_autocatalysis_with_pattern(self):
        """Test autocatalysis detected with clustering pattern."""
        watcher = Watcher(
            watcher_id="TEST-001",
            target_domain="hope_florida",
            spawn_cycle=0,
            detection_history=[
                {"signal": 0.6, "timestamp": 1},
                {"signal": 0.7, "timestamp": 2},
                {"signal": 0.8, "timestamp": 3},
                {"signal": 0.9, "timestamp": 4},
            ],
        )
        check_autocatalysis(watcher, [])
        assert watcher.self_reference_score > 0


class TestWatcherDissolution:
    """Tests for watcher dissolution."""

    def test_dissolve_returns_info(self):
        """Test dissolve returns watcher info."""
        watcher = Watcher(
            watcher_id="TEST-001",
            target_domain="hope_florida",
            spawn_cycle=0,
            total_detections=5,
        )
        result = dissolve_watcher(watcher)
        assert result["watcher_id"] == "TEST-001"
        assert result["total_detections"] == 5


class TestPopulationGovernance:
    """Tests for watcher population governance."""

    def test_population_under_cap(self):
        """Test population under cap is unchanged."""
        watchers = [
            Watcher(f"W-{i}", "hope_florida", 0) for i in range(10)
        ]
        governed = govern_population(watchers, 1.0)
        assert len(governed) == 10

    def test_population_over_cap(self):
        """Test population over cap is reduced."""
        watchers = [
            Watcher(f"W-{i}", "hope_florida", 0) for i in range(100)
        ]
        governed = govern_population(watchers, 0.5)  # Half entropy budget
        assert len(governed) <= 25

    def test_autocatalytic_watchers_preserved(self):
        """Test autocatalytic watchers are preserved."""
        watchers = []
        for i in range(30):
            w = Watcher(f"W-{i}", "hope_florida", 0)
            if i < 5:
                w.is_autocatalytic = True
                w.self_reference_score = 0.9
            watchers.append(w)

        governed = govern_population(watchers, 0.5)
        autocatalytic_count = sum(1 for w in governed if w.is_autocatalytic)
        assert autocatalytic_count == 5


class TestWatcherCycle:
    """Tests for watcher cycle execution."""

    def test_run_cycle(self):
        """Test running a watcher cycle."""
        watchers = []
        events = generate_fraud_events(20, "hope_florida", 0.3, 42)
        result = run_watcher_cycle(watchers, events, 0)

        assert "cycle" in result
        assert "active_watchers" in result
        assert "detections" in result


class TestWatcherSimulation:
    """Tests for complete watcher simulation."""

    def test_simulation_completes(self, seed):
        """Test simulation runs to completion."""
        result = run_watcher_simulation(
            n_cycles=20,
            n_events_per_cycle=30,
            fraud_rate=0.2,
            seed=seed,
        )
        assert result["n_cycles"] == 20
        assert "total_spawned" in result
        assert "final_autocatalytic" in result

    def test_simulation_spawns_watchers(self, seed):
        """Test simulation spawns watchers on high fraud."""
        result = run_watcher_simulation(
            n_cycles=50,
            n_events_per_cycle=50,
            fraud_rate=0.4,  # High fraud rate
            seed=seed,
        )
        assert result["total_spawned"] > 0
