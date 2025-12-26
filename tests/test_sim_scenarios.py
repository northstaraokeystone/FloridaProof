"""Tests for simulation scenarios."""

import pytest
from src.sim import (
    SimConfig,
    SimState,
    SimResult,
    run_simulation,
    run_scenario,
    run_all_scenarios,
    run_domain_detection,
    simulate_cycle,
    validate_constraints,
    compute_aggregate_detection,
)


class TestSimConfig:
    """Tests for simulation configuration."""

    def test_default_config(self):
        """Test default configuration values."""
        config = SimConfig()
        assert config.n_cycles == 10000
        assert config.random_seed == 42
        assert "hope_florida" in config.fraud_rates

    def test_custom_config(self):
        """Test custom configuration."""
        config = SimConfig(n_cycles=100, random_seed=123)
        assert config.n_cycles == 100
        assert config.random_seed == 123


class TestDomainDetection:
    """Tests for domain detection."""

    def test_hope_florida_detection(self, sim_config):
        """Test Hope Florida detection runs."""
        result = run_domain_detection("hope_florida", sim_config, 42)
        assert "detection_rate" in result
        assert "receipts" in result

    def test_insurance_detection(self, sim_config):
        """Test insurance detection runs."""
        result = run_domain_detection("insurance", sim_config, 42)
        assert "detection_rate" in result or "influence_score" in result

    def test_sheriff_detection(self, sim_config):
        """Test sheriff detection runs."""
        result = run_domain_detection("sheriff", sim_config, 42)
        assert "detection_rate" in result

    def test_corporate_detection(self, sim_config):
        """Test corporate detection runs."""
        result = run_domain_detection("corporate", sim_config, 42)
        assert "detection_rate" in result

    def test_pandemic_detection(self, sim_config):
        """Test pandemic detection runs."""
        result = run_domain_detection("pandemic", sim_config, 42)
        assert "detection_rate" in result


class TestSimulation:
    """Tests for simulation execution."""

    def test_simulation_completes(self, sim_config):
        """Test simulation runs to completion."""
        sim_config.n_cycles = 10
        result = run_simulation(sim_config)
        assert isinstance(result, SimResult)
        assert result.final_state.cycle > 0

    def test_simulation_generates_receipts(self, sim_config):
        """Test simulation generates receipts."""
        sim_config.n_cycles = 5
        result = run_simulation(sim_config)
        assert len(result.final_state.receipt_ledger) > 0

    def test_simulation_tracks_detection(self, sim_config):
        """Test simulation tracks detection rates."""
        sim_config.n_cycles = 10
        result = run_simulation(sim_config)
        assert len(result.detection_rates) > 0


class TestScenarios:
    """Tests for scenario execution."""

    def test_baseline_scenario(self, sim_config):
        """Test BASELINE scenario."""
        result = run_scenario("BASELINE", sim_config)
        assert result.scenario == "BASELINE"
        # BASELINE should pass with default config
        assert "cycles_completed" in result.details

    def test_stress_scenario(self, sim_config):
        """Test STRESS scenario."""
        result = run_scenario("STRESS", sim_config)
        assert result.scenario == "STRESS"
        assert "high_fraud_handled" in result.details

    def test_genesis_scenario(self, sim_config):
        """Test GENESIS scenario."""
        result = run_scenario("GENESIS", sim_config)
        assert result.scenario == "GENESIS"
        assert "watchers_spawned" in result.details

    def test_cascade_scenario(self, sim_config):
        """Test CASCADE scenario."""
        result = run_scenario("CASCADE", sim_config)
        assert result.scenario == "CASCADE"
        assert "cascades_detected" in result.details

    def test_pressure_scenario(self, sim_config):
        """Test PRESSURE scenario."""
        result = run_scenario("PRESSURE", sim_config)
        assert result.scenario == "PRESSURE"
        assert "alpha_50" in result.details

    def test_godel_scenario(self, sim_config):
        """Test GODEL scenario."""
        result = run_scenario("GODEL", sim_config)
        assert result.scenario == "GODEL"
        assert "zero_fraud_handled" in result.details

    def test_unknown_scenario(self, sim_config):
        """Test unknown scenario handling."""
        result = run_scenario("UNKNOWN", sim_config)
        assert result.passed is False


class TestAllScenarios:
    """Tests for running all scenarios."""

    def test_all_scenarios_run(self, sim_config):
        """Test all scenarios execute."""
        result = run_all_scenarios(sim_config)
        assert "scenarios" in result
        assert len(result["scenarios"]) == 6

    def test_all_scenarios_report(self, sim_config):
        """Test all scenarios report status."""
        result = run_all_scenarios(sim_config)
        assert "all_passed" in result
        assert "passed_count" in result


class TestConstraintValidation:
    """Tests for constraint validation."""

    def test_validate_passing(self):
        """Test validation with passing rates."""
        state = SimState()
        detection_rates = {
            "hope_florida": 0.95,
            "insurance": 0.90,
            "sheriff": 0.92,
            "corporate": 0.88,
            "pandemic": 0.97,
        }
        violations = validate_constraints(state, detection_rates)
        assert len(violations) == 0

    def test_validate_failing(self):
        """Test validation with failing rates."""
        state = SimState()
        detection_rates = {
            "hope_florida": 0.50,  # Below threshold
            "insurance": 0.90,
            "sheriff": 0.92,
            "corporate": 0.88,
            "pandemic": 0.97,
        }
        violations = validate_constraints(state, detection_rates)
        assert len(violations) > 0


class TestAggregateDetection:
    """Tests for aggregate detection computation."""

    def test_aggregate_empty(self):
        """Test aggregate with empty state."""
        state = SimState()
        result = compute_aggregate_detection(state)
        assert result["domains_analyzed"] == 0

    def test_aggregate_with_data(self, sim_config):
        """Test aggregate with simulation data."""
        sim_config.n_cycles = 5
        sim_result = run_simulation(sim_config)
        result = compute_aggregate_detection(sim_result.final_state)
        assert result["domains_analyzed"] > 0
        assert "average_detection" in result
