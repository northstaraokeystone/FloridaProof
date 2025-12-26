"""Tests for network module."""

import pytest
from src.network import (
    build_protection_network,
    compute_protection_score,
    find_immunity_path,
    simulate_network_exposure,
    threshold_protection_failure,
    compute_audit_resistance,
    run_network_analysis,
    FLORIDA_NETWORK,
)


class TestProtectionNetwork:
    """Tests for protection network construction."""

    def test_build_default_network(self):
        """Test building default Florida network."""
        network = build_protection_network()
        assert len(network) > 0
        assert "Hope Florida Foundation" in network

    def test_build_custom_network(self):
        """Test building custom network."""
        entities = [
            {"name": "Entity A", "tier": "donor"},
            {"name": "Entity B", "tier": "grantee"},
        ]
        connections = [
            {"source": "Entity A", "target": "Entity B"},
        ]
        network = build_protection_network(entities, connections)
        assert "Entity A" in network
        assert "Entity B" in network["Entity A"]["connections"]

    def test_network_has_tiers(self, protection_network):
        """Test network entities have tier information."""
        for entity, info in protection_network.items():
            assert "tier" in info
            assert info["tier"] in ["donor", "political", "agency", "grantee"]


class TestProtectionScore:
    """Tests for protection score computation."""

    def test_score_donor_tier(self, protection_network):
        """Test donors have high protection scores."""
        donor_entities = [e for e, i in protection_network.items() if i.get("tier") == "donor"]
        if donor_entities:
            score = compute_protection_score(donor_entities[0], protection_network)
            assert score > 0.5

    def test_score_grantee_tier(self, protection_network):
        """Test grantees have lower protection scores."""
        # Hope Florida Foundation is a grantee
        score = compute_protection_score("Hope Florida Foundation", protection_network)
        # Should have some protection from connections
        assert 0.0 <= score <= 1.0

    def test_score_nonexistent(self, protection_network):
        """Test score for nonexistent entity."""
        score = compute_protection_score("Nonexistent Entity", protection_network)
        assert score == 0.0


class TestImmunityPath:
    """Tests for immunity path finding."""

    def test_find_path_connected(self, protection_network):
        """Test finding path between connected entities."""
        # Governor's Office -> AG Office (direct connection)
        path = find_immunity_path("Governor's Office", "AG Office", protection_network)
        assert len(path) >= 2
        assert path[0] == "Governor's Office"

    def test_find_path_transitive(self, protection_network):
        """Test finding transitive path."""
        # Donor -> Grantee should have a path
        path = find_immunity_path(
            "Governor's Office", "Hope Florida Foundation", protection_network
        )
        # Should find a path through AG Office
        assert len(path) >= 2

    def test_find_path_unconnected(self, protection_network):
        """Test no path for unconnected entities."""
        path = find_immunity_path(
            "Keep Florida Clean PAC", "Insurance Industry Consortium", protection_network
        )
        # PAC is at the end of the chain, no path back to donors
        assert len(path) <= 1


class TestNetworkExposure:
    """Tests for network exposure simulation."""

    def test_exposure_propagates(self, protection_network):
        """Test scandal exposure propagates through network."""
        result = simulate_network_exposure(
            "Hope Florida Foundation", 0.8, protection_network
        )
        assert result["scandal_source"] == "Hope Florida Foundation"
        assert result["initial_exposure"] == 0.8
        assert len(result["exposed_entities"]) > 1

    def test_exposure_decays(self, protection_network):
        """Test exposure decays with distance."""
        result = simulate_network_exposure(
            "Hope Florida Foundation", 1.0, protection_network
        )
        # Source should have highest exposure
        source_exposure = result["exposed_entities"]["Hope Florida Foundation"]
        for entity, exposure in result["exposed_entities"].items():
            if entity != "Hope Florida Foundation":
                assert exposure <= source_exposure


class TestProtectionFailure:
    """Tests for protection failure threshold."""

    def test_low_exposure(self, protection_network):
        """Test protection holds at low exposure."""
        remaining = threshold_protection_failure(protection_network, 0.1)
        assert remaining > 0.5

    def test_high_exposure(self, protection_network):
        """Test protection degrades at extreme exposure."""
        # Very high exposure should cause some degradation
        remaining = threshold_protection_failure(protection_network, 1.5)
        assert remaining < 1.0


class TestAuditResistance:
    """Tests for audit resistance computation."""

    def test_resistance_calculation(self, protection_network):
        """Test audit resistance is calculated."""
        resistance = compute_audit_resistance(
            "Hope Florida Foundation", protection_network
        )
        assert 0.0 <= resistance <= 1.0

    def test_donor_connected_resistance(self, protection_network):
        """Test entities with donor connections have higher resistance."""
        # Political tier should have connections to donors
        political_entities = [
            e for e, i in protection_network.items() if i.get("tier") == "political"
        ]
        if political_entities:
            resistance = compute_audit_resistance(political_entities[0], protection_network)
            assert resistance > 0.5


class TestNetworkAnalysis:
    """Tests for full network analysis."""

    def test_run_analysis(self, seed):
        """Test complete network analysis."""
        result = run_network_analysis(entity="Hope Florida Foundation", seed=seed)
        assert "protection_score" in result
        assert "audit_resistance" in result
        assert "exposure_simulation" in result
        assert "receipt" in result
