"""Tests for pressure module."""

import pytest
from src.pressure import (
    simulate_audit_disruption,
    simulate_probe_halt,
    simulate_non_cooperation,
    compute_alpha,
    simulate_pressure_scenario,
    run_pressure_sweep,
    run_pressure_analysis,
)


class TestAuditDisruption:
    """Tests for audit disruption simulation."""

    def test_disruption_zero(self, seed):
        """Test zero disruption has no effect."""
        result = simulate_audit_disruption(0.0, n_investigators=10, seed=seed)
        assert result["investigators_reassigned"] == 0
        assert result["remaining_detection_capacity"] == 1.0

    def test_disruption_high(self, seed):
        """Test high disruption reduces capacity."""
        result = simulate_audit_disruption(0.9, n_investigators=10, seed=seed)
        assert result["investigators_reassigned"] > 0
        assert result["remaining_detection_capacity"] < 1.0

    def test_disruption_timeline_extension(self, seed):
        """Test disruption causes timeline extension."""
        result = simulate_audit_disruption(0.5, seed=seed)
        assert result["timeline_extension_days"] > 0


class TestProbeHalt:
    """Tests for probe halt simulation."""

    def test_probe_halt_zero_pressure(self, seed):
        """Test probe not halted at zero pressure."""
        result = simulate_probe_halt("PROBE-001", 0.0, seed=seed)
        assert result["is_halted"] is False

    def test_probe_halt_high_pressure(self, seed):
        """Test probe likely halted at high pressure."""
        # With high pressure, most probes should be halted
        halt_count = 0
        for i in range(10):
            result = simulate_probe_halt(f"PROBE-{i}", 0.95, seed=seed + i)
            if result["is_halted"]:
                halt_count += 1
        assert halt_count > 5  # Most should be halted

    def test_probe_halt_has_reason(self, seed):
        """Test halted probe has halt reason."""
        result = simulate_probe_halt("PROBE-001", 1.0, seed=seed)
        if result["is_halted"]:
            assert result["halt_reason"] is not None


class TestNonCooperation:
    """Tests for subpoena non-cooperation simulation."""

    def test_cooperation_zero_pressure(self, seed):
        """Test full cooperation at zero pressure."""
        subpoenas = [f"SUB-{i}" for i in range(10)]
        results = simulate_non_cooperation(subpoenas, 0.0, seed=seed)
        ignored_count = sum(1 for r in results if r["ignored"])
        assert ignored_count == 0

    def test_non_cooperation_high_pressure(self, seed):
        """Test non-cooperation at high pressure."""
        subpoenas = [f"SUB-{i}" for i in range(10)]
        results = simulate_non_cooperation(subpoenas, 0.9, seed=seed)
        ignored_count = sum(1 for r in results if r["ignored"])
        assert ignored_count > 5


class TestAlphaComputation:
    """Tests for NEURON alpha computation."""

    def test_alpha_no_degradation(self):
        """Test alpha = 1.0 when no degradation."""
        alpha = compute_alpha(0.92, 0.92)
        assert alpha == 1.0

    def test_alpha_partial_degradation(self):
        """Test alpha reflects partial degradation."""
        alpha = compute_alpha(1.0, 0.7)
        assert alpha == 0.7

    def test_alpha_zero_baseline(self):
        """Test alpha handles zero baseline."""
        alpha = compute_alpha(0.0, 0.5)
        assert alpha == 0.0

    def test_alpha_bounded(self):
        """Test alpha is bounded to [0, 1]."""
        alpha = compute_alpha(0.5, 1.0)
        assert alpha <= 1.0


class TestPressureScenario:
    """Tests for pressure scenario simulation."""

    def test_scenario_baseline(self, seed):
        """Test baseline scenario (zero pressure)."""
        system_state = {"detection_rate": 0.92}
        result = simulate_pressure_scenario(system_state, 0.0, seed=seed)
        assert result["alpha"] >= 0.9

    def test_scenario_high_pressure(self, seed):
        """Test high pressure scenario."""
        system_state = {"detection_rate": 0.92}
        result = simulate_pressure_scenario(system_state, 0.8, seed=seed)
        assert result["alpha"] < 1.0
        assert result["detection_under_pressure"] < 0.92

    def test_scenario_has_state(self, seed):
        """Test scenario includes pressure state."""
        system_state = {"detection_rate": 0.92}
        result = simulate_pressure_scenario(system_state, 0.5, seed=seed)
        assert result["pressure_state"] is not None


class TestPressureSweep:
    """Tests for pressure sweep."""

    def test_sweep_multiple_levels(self, seed):
        """Test sweep covers multiple pressure levels."""
        result = run_pressure_sweep(seed=seed)
        assert len(result["results"]) == 5  # Default 5 levels

    def test_sweep_alpha_degrades(self, seed):
        """Test alpha degrades with pressure."""
        result = run_pressure_sweep(seed=seed)
        alphas = [r["alpha"] for r in result["results"]]
        # Should generally decrease with pressure
        assert alphas[0] >= alphas[-1]

    def test_sweep_slo_check(self, seed):
        """Test sweep checks SLO compliance."""
        result = run_pressure_sweep(seed=seed)
        assert "slo_50_passed" in result
        assert "slo_75_passed" in result


class TestPressureAnalysis:
    """Tests for complete pressure analysis."""

    def test_analysis_completes(self, seed):
        """Test analysis runs to completion."""
        result = run_pressure_analysis(seed=seed)
        assert "alpha_50" in result
        assert "alpha_75" in result
        assert "receipts" in result

    def test_analysis_slo_compliance(self, seed):
        """Test analysis reports SLO compliance."""
        result = run_pressure_analysis(seed=seed)
        # With default parameters, should pass SLOs
        assert "slo_passed" in result
