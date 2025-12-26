"""Protection Network Mapping.

Maps protection networks that shield fraud from detection.
Pattern: Donor tier -> Political tier -> Agency tier -> Grantee tier.
"""

from dataclasses import dataclass, field
from typing import Any

from .core import TENANT_ID, emit_receipt


@dataclass
class NetworkEntity:
    """Represents an entity in the protection network."""

    entity_id: str
    name: str
    tier: str  # donor, political, agency, grantee
    connections: list = field(default_factory=list)
    protection_score: float = 0.0


@dataclass
class NetworkConnection:
    """Represents a connection between entities."""

    source: str
    target: str
    connection_type: str  # donation, appointment, oversight, grant
    strength: float = 1.0


# Florida protection network structure
FLORIDA_NETWORK = {
    # Donor tier
    "Insurance Industry Consortium": {
        "tier": "donor",
        "connects_to": ["Governor's Office", "Legislature Leadership"],
    },
    "Publix/Jenkins Family": {
        "tier": "donor",
        "connects_to": ["Governor's Office", "FL GOP"],
    },
    "FPL/NextEra": {
        "tier": "donor",
        "connects_to": ["OIR", "Legislature Leadership"],
    },
    "U.S. Sugar": {
        "tier": "donor",
        "connects_to": ["SFWMD", "Legislature Leadership"],
    },
    # Political tier
    "Governor's Office": {
        "tier": "political",
        "connects_to": ["AG Office", "DEO", "AHCA", "OIR"],
    },
    "AG Office": {
        "tier": "political",
        "connects_to": ["Hope Florida Foundation", "Keep Florida Clean PAC"],
    },
    "Legislature Leadership": {
        "tier": "political",
        "connects_to": ["Committee Chairs", "Agency Heads"],
    },
    "County Sheriffs": {
        "tier": "political",
        "connects_to": ["Sheriff Contractors", "County Audit Committees"],
    },
    # Agency tier
    "DEO": {
        "tier": "agency",
        "connects_to": ["Unemployment Programs"],
    },
    "OIR": {
        "tier": "agency",
        "connects_to": ["Insurance Rate Approvals"],
    },
    "AHCA": {
        "tier": "agency",
        "connects_to": ["Medicaid Oversight", "Hope Florida Foundation"],
    },
    "County Audit Committees": {
        "tier": "agency",
        "connects_to": ["Sheriff Contractors"],
    },
    # Grantee tier
    "Hope Florida Foundation": {
        "tier": "grantee",
        "connects_to": ["Secure Florida's Future", "Save Our Society from Drugs"],
    },
    "Secure Florida's Future": {
        "tier": "grantee",
        "connects_to": ["Keep Florida Clean PAC"],
    },
    "Save Our Society from Drugs": {
        "tier": "grantee",
        "connects_to": ["Keep Florida Clean PAC"],
    },
    "Keep Florida Clean PAC": {
        "tier": "grantee",
        "connects_to": [],
    },
    "Sheriff Contractors": {
        "tier": "grantee",
        "connects_to": [],
    },
}

# Tier protection multipliers
TIER_PROTECTION = {
    "donor": 0.9,  # High protection
    "political": 0.85,
    "agency": 0.7,
    "grantee": 0.5,  # Lower protection
}


def build_protection_network(
    entities: list[dict] | None = None, connections: list[dict] | None = None
) -> dict:
    """Construct protection network graph."""
    if entities is None:
        # Use default Florida network
        network = {}
        for entity_name, info in FLORIDA_NETWORK.items():
            network[entity_name] = {
                "tier": info["tier"],
                "connections": info["connects_to"],
                "protection_base": TIER_PROTECTION[info["tier"]],
            }
        return network

    # Build from provided entities
    network = {}
    for entity in entities:
        network[entity["name"]] = {
            "tier": entity.get("tier", "grantee"),
            "connections": [],
            "protection_base": TIER_PROTECTION.get(entity.get("tier", "grantee"), 0.5),
        }

    if connections:
        for conn in connections:
            source = conn["source"]
            if source in network:
                network[source]["connections"].append(conn["target"])

    return network


def compute_protection_score(entity: str, network: dict) -> float:
    """Calculate how protected this entity is from audit."""
    if entity not in network:
        return 0.0

    entity_info = network[entity]
    base_score = entity_info.get("protection_base", 0.5)

    # Add protection from connections to higher tiers
    connection_bonus = 0.0
    for connected in entity_info.get("connections", []):
        if connected in network:
            connected_tier = network[connected].get("tier", "grantee")
            tier_value = {"donor": 0.15, "political": 0.1, "agency": 0.05, "grantee": 0.02}
            connection_bonus += tier_value.get(connected_tier, 0)

    # Check reverse connections (who protects this entity)
    for other_entity, other_info in network.items():
        if entity in other_info.get("connections", []):
            other_tier = other_info.get("tier", "grantee")
            tier_value = {"donor": 0.2, "political": 0.15, "agency": 0.08, "grantee": 0.03}
            connection_bonus += tier_value.get(other_tier, 0)

    return min(1.0, base_score + connection_bonus)


def find_immunity_path(donor: str, grantee: str, network: dict) -> list:
    """Trace protection pathway from donor to grantee."""
    visited = set()
    path = []

    def dfs(current: str, target: str) -> bool:
        if current == target:
            path.append(current)
            return True

        visited.add(current)
        path.append(current)

        if current not in network:
            path.pop()
            return False

        for connected in network[current].get("connections", []):
            if connected not in visited:
                if dfs(connected, target):
                    return True

        path.pop()
        return False

    dfs(donor, grantee)
    return path


def simulate_network_exposure(
    scandal_entity: str, exposure_level: float, network: dict
) -> dict:
    """Model how scandal propagates through network."""
    exposed_entities = {scandal_entity: exposure_level}
    visited = {scandal_entity}

    # Find all connected entities and propagate exposure
    def propagate(entity: str, current_exposure: float):
        if current_exposure < 0.05:  # Threshold
            return

        if entity not in network:
            return

        # Exposure to connected entities (forward)
        for connected in network[entity].get("connections", []):
            if connected not in visited:
                visited.add(connected)
                # Exposure decays based on tier protection
                connected_protection = compute_protection_score(connected, network)
                propagated_exposure = current_exposure * (1 - connected_protection * 0.5)
                exposed_entities[connected] = propagated_exposure
                propagate(connected, propagated_exposure * 0.7)

        # Exposure to entities that connect to this one (backward)
        for other_entity, other_info in network.items():
            if entity in other_info.get("connections", []) and other_entity not in visited:
                visited.add(other_entity)
                other_protection = compute_protection_score(other_entity, network)
                propagated_exposure = current_exposure * (1 - other_protection * 0.7)
                exposed_entities[other_entity] = propagated_exposure
                propagate(other_entity, propagated_exposure * 0.5)

    propagate(scandal_entity, exposure_level)

    return {
        "scandal_source": scandal_entity,
        "initial_exposure": exposure_level,
        "exposed_entities": exposed_entities,
        "total_exposed": len(exposed_entities),
        "max_propagation": max(exposed_entities.values()) if exposed_entities else 0,
    }


def threshold_protection_failure(network: dict, exposure: float) -> float:
    """Calculate at what exposure level protection collapses."""
    # Average protection across network
    total_protection = sum(
        compute_protection_score(entity, network) for entity in network
    )
    avg_protection = total_protection / len(network) if network else 0

    # Protection fails when exposure exceeds protection
    failure_threshold = avg_protection * 1.2  # 20% buffer

    if exposure > failure_threshold:
        # Calculate how much protection remains
        remaining = max(0, (failure_threshold * 2 - exposure) / failure_threshold)
        return remaining

    return 1.0  # Full protection still intact


def compute_audit_resistance(entity: str, network: dict) -> float:
    """Calculate how resistant entity is to audit."""
    protection = compute_protection_score(entity, network)

    # Find immunity paths from donors
    donor_paths = 0
    for other_entity, info in network.items():
        if info.get("tier") == "donor":
            path = find_immunity_path(other_entity, entity, network)
            if len(path) > 1:
                donor_paths += 1

    # More donor connections = more audit resistance
    path_bonus = min(0.2, donor_paths * 0.05)

    return min(1.0, protection + path_bonus)


def emit_network_receipt(entity: str, network: dict) -> dict:
    """Emit receipt for network analysis."""
    protection = compute_protection_score(entity, network)
    audit_resistance = compute_audit_resistance(entity, network)

    # Find connected donors and politicians
    donors = []
    politicians = []
    for other_entity, info in network.items():
        connections = info.get("connections", [])
        if entity in connections:
            if info.get("tier") == "donor":
                donors.append(other_entity)
            elif info.get("tier") == "political":
                politicians.append(other_entity)

    # Calculate exposure threshold
    exposure_result = simulate_network_exposure(entity, 0.8, network)
    exposure_threshold = threshold_protection_failure(network, 0.8)

    return emit_receipt(
        "network",
        {
            "tenant_id": TENANT_ID,
            "entity": entity,
            "protection_score": protection,
            "connected_donors": donors,
            "connected_politicians": politicians,
            "audit_resistance": audit_resistance,
            "exposure_threshold": exposure_threshold,
        },
    )


def run_network_analysis(entity: str | None = None, seed: int = 42) -> dict:
    """Run network analysis for specified entity."""
    network = build_protection_network()

    if entity is None:
        entity = "Hope Florida Foundation"

    protection = compute_protection_score(entity, network)
    audit_resistance = compute_audit_resistance(entity, network)
    exposure_sim = simulate_network_exposure(entity, 0.8, network)
    failure_threshold = threshold_protection_failure(network, 0.8)

    receipt = emit_network_receipt(entity, network)

    return {
        "network": network,
        "entity": entity,
        "protection_score": protection,
        "audit_resistance": audit_resistance,
        "exposure_simulation": exposure_sim,
        "failure_threshold": failure_threshold,
        "receipt": receipt,
    }
