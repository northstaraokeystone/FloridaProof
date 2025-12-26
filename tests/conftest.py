"""Pytest configuration and fixtures for FloridaProof tests."""

import pytest
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


@pytest.fixture
def seed():
    """Standard test seed for reproducibility."""
    return 42


@pytest.fixture
def sim_config():
    """Default simulation configuration for tests."""
    from src.sim import SimConfig

    return SimConfig(
        n_cycles=10,
        n_settlements=20,
        n_contracts=50,
        random_seed=42,
    )


@pytest.fixture
def cascade_graph():
    """Pre-built cascade graph for tests."""
    from src.cascade import build_cascade_graph, CASCADE_WEIGHTS

    return build_cascade_graph(CASCADE_WEIGHTS)


@pytest.fixture
def protection_network():
    """Pre-built protection network for tests."""
    from src.network import build_protection_network

    return build_protection_network()


@pytest.fixture
def sample_settlement():
    """Sample settlement flow for tests."""
    from src.domains.hope_florida import SettlementFlow

    return SettlementFlow(
        settlement_id="TEST-001",
        source_amount=67_000_000,
        foundation_allocation=10_000_000,
        downstream_grants=[
            {"entity": "Secure Florida's Future", "amount": 5_000_000},
            {"entity": "Save Our Society from Drugs", "amount": 5_000_000},
        ],
        timestamp_days_from_settlement=30,
        entity_overlap_score=0.3,
    )


@pytest.fixture
def sample_contract():
    """Sample sheriff contract for tests."""
    from src.domains.sheriff import SheriffContract

    return SheriffContract(
        contract_id="CONTRACT-TEST-001",
        sheriff_office="Lee County",
        contractor="CONTRACTOR-001",
        monthly_amount=5700,
        contract_type="consulting",
        deliverables_documented=False,
        personal_tie_detected=True,
    )
