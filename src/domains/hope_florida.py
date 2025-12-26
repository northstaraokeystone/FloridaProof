"""Hope Florida Fund Routing Detection.

Detects settlement -> foundation -> PAC fund routing patterns.

The Hope Florida Topology (documented):
- $67M Centene Medicaid settlement
- $10M -> Hope Florida Foundation
- $5M -> Secure Florida's Future (FL Chamber linked)
- $5M -> Save Our Society from Drugs
- Both -> Keep Florida Clean PAC (Uthmeier chaired, opposed Amendment 3)
"""

import random
from dataclasses import dataclass, field
from typing import Any

from ..core import TENANT_ID, dual_hash, emit_receipt


@dataclass
class SettlementFlow:
    """Represents a settlement fund flow."""

    settlement_id: str
    source_amount: float
    foundation_allocation: float
    downstream_grants: list = field(default_factory=list)
    timestamp_days_from_settlement: int = 0
    entity_overlap_score: float = 0.0
    is_fraud: bool = False
    fraud_type: str | None = None


# Key entities in the Hope Florida topology
HOPE_FLORIDA_ENTITIES = {
    "Hope Florida Foundation": {"type": "foundation", "key_figure": "Casey DeSantis"},
    "Secure Florida's Future": {"type": "nonprofit", "linked_to": "FL Chamber"},
    "Save Our Society from Drugs": {"type": "nonprofit", "grant_amount": 5000000},
    "Keep Florida Clean PAC": {"type": "pac", "chaired_by": "James Uthmeier"},
}

# Entity graph for connection detection
ENTITY_GRAPH = {
    "Hope Florida Foundation": ["Secure Florida's Future", "Save Our Society from Drugs"],
    "Secure Florida's Future": ["Keep Florida Clean PAC", "FL Chamber"],
    "Save Our Society from Drugs": ["Keep Florida Clean PAC"],
    "Keep Florida Clean PAC": [],
}


def generate_settlement_routing(
    n_settlements: int, fraud_rate: float, seed: int
) -> list[SettlementFlow]:
    """Generate synthetic settlement -> foundation flows."""
    rng = random.Random(seed)
    settlements = []

    for i in range(n_settlements):
        source_amount = rng.uniform(1_000_000, 100_000_000)
        foundation_alloc = source_amount * rng.uniform(0.05, 0.25)

        # Generate downstream grants
        n_grants = rng.randint(1, 5)
        grants = []
        remaining = foundation_alloc
        for j in range(n_grants):
            grant_amount = remaining * rng.uniform(0.1, 0.5)
            remaining -= grant_amount
            entity = rng.choice(list(HOPE_FLORIDA_ENTITIES.keys()))
            grants.append({"entity": entity, "amount": grant_amount})

        flow = SettlementFlow(
            settlement_id=f"SETTLE-{seed}-{i:04d}",
            source_amount=source_amount,
            foundation_allocation=foundation_alloc,
            downstream_grants=grants,
            timestamp_days_from_settlement=rng.randint(1, 180),
            entity_overlap_score=rng.uniform(0.0, 0.5),
        )
        settlements.append(flow)

    return inject_routing_fraud(settlements, fraud_rate, rng)


def inject_routing_fraud(
    settlements: list[SettlementFlow], fraud_rate: float, rng: random.Random | None = None
) -> list[SettlementFlow]:
    """Mark settlements with improper routing patterns."""
    if rng is None:
        rng = random.Random()

    fraud_types = ["routing", "timing", "entity_overlap", "pac_connection"]

    for settlement in settlements:
        if rng.random() < fraud_rate:
            settlement.is_fraud = True
            settlement.fraud_type = rng.choice(fraud_types)

            # Inject fraud indicators based on type
            if settlement.fraud_type == "routing":
                # Complex routing through multiple entities
                settlement.downstream_grants = [
                    {"entity": "Hope Florida Foundation", "amount": settlement.foundation_allocation * 0.3},
                    {"entity": "Secure Florida's Future", "amount": settlement.foundation_allocation * 0.35},
                    {"entity": "Keep Florida Clean PAC", "amount": settlement.foundation_allocation * 0.35},
                ]
            elif settlement.fraud_type == "timing":
                # Suspiciously quick routing (within 30 days)
                settlement.timestamp_days_from_settlement = rng.randint(1, 30)
            elif settlement.fraud_type == "entity_overlap":
                # High entity overlap (board members, legal counsel)
                settlement.entity_overlap_score = rng.uniform(0.7, 1.0)
            elif settlement.fraud_type == "pac_connection":
                # Direct PAC connection detected
                settlement.downstream_grants.append(
                    {"entity": "Keep Florida Clean PAC", "amount": settlement.foundation_allocation * 0.2}
                )

    return settlements


def detect_foundation_pac_link(flow: SettlementFlow, entity_graph: dict | None = None) -> dict:
    """Check if foundation grant connects to political PAC."""
    if entity_graph is None:
        entity_graph = ENTITY_GRAPH

    pac_linked = False
    pac_entities = []
    path = []

    for grant in flow.downstream_grants:
        entity = grant["entity"]
        if "PAC" in entity:
            pac_linked = True
            pac_entities.append(entity)
            path.append(entity)
        elif entity in entity_graph:
            # Check one level deep for PAC connections
            for connected in entity_graph.get(entity, []):
                if "PAC" in connected:
                    pac_linked = True
                    pac_entities.append(connected)
                    path.extend([entity, connected])

    return {
        "pac_linked": pac_linked,
        "pac_entities": list(set(pac_entities)),
        "path": path,
        "link_strength": len(pac_entities) / max(len(flow.downstream_grants), 1),
    }


def score_routing_risk(flow: SettlementFlow) -> float:
    """Risk score 0-1 based on routing complexity, timing, entity overlap."""
    risk = 0.0

    # Timing risk: faster routing = higher risk
    if flow.timestamp_days_from_settlement < 30:
        risk += 0.3
    elif flow.timestamp_days_from_settlement < 60:
        risk += 0.15

    # Entity overlap risk
    risk += flow.entity_overlap_score * 0.3

    # Routing complexity risk
    if len(flow.downstream_grants) > 3:
        risk += 0.2

    # PAC connection risk
    pac_link = detect_foundation_pac_link(flow)
    if pac_link["pac_linked"]:
        risk += 0.2 + pac_link["link_strength"] * 0.1

    return min(1.0, risk)


def check_uthmeier_connection(entity: str, graph: dict | None = None) -> bool:
    """Special check for AG/Chief-of-Staff involvement."""
    if graph is None:
        graph = ENTITY_GRAPH

    # Direct check
    if entity == "Keep Florida Clean PAC":
        return True

    # Check connections
    for connected in graph.get(entity, []):
        if connected == "Keep Florida Clean PAC":
            return True

    return False


def detect_fraud(flow: SettlementFlow) -> tuple[bool, dict]:
    """Main fraud detection function for Hope Florida flows."""
    risk_score = score_routing_risk(flow)
    pac_link = detect_foundation_pac_link(flow)
    uthmeier_linked = any(
        check_uthmeier_connection(g["entity"]) for g in flow.downstream_grants
    )

    # Detection thresholds
    fraud_detected = (
        risk_score > 0.6
        or (pac_link["pac_linked"] and flow.timestamp_days_from_settlement < 90)
        or (flow.entity_overlap_score > 0.7 and pac_link["pac_linked"])
    )

    details = {
        "risk_score": risk_score,
        "pac_link": pac_link,
        "uthmeier_linked": uthmeier_linked,
        "timing_flag": flow.timestamp_days_from_settlement < 30,
        "overlap_flag": flow.entity_overlap_score > 0.7,
    }

    return fraud_detected, details


def emit_hope_florida_receipt(flow: SettlementFlow, detection_result: tuple[bool, dict]) -> dict:
    """Emit receipt for Hope Florida fraud detection."""
    fraud_detected, details = detection_result

    return emit_receipt(
        "hope_florida",
        {
            "tenant_id": TENANT_ID,
            "settlement_id": flow.settlement_id,
            "source_settlement": flow.source_amount,
            "foundation_allocation": flow.foundation_allocation,
            "downstream_grants": flow.downstream_grants,
            "pac_connection_detected": details["pac_link"]["pac_linked"],
            "routing_risk_score": details["risk_score"],
            "uthmeier_linked": details["uthmeier_linked"],
            "fraud_detected": fraud_detected,
            "fraud_type": flow.fraud_type if fraud_detected else None,
        },
    )


def run_detection(
    n_settlements: int = 100, fraud_rate: float = 0.15, seed: int = 42
) -> dict:
    """Run Hope Florida fraud detection on generated data."""
    settlements = generate_settlement_routing(n_settlements, fraud_rate, seed)

    true_positives = 0
    false_positives = 0
    true_negatives = 0
    false_negatives = 0
    receipts = []

    for flow in settlements:
        detected, details = detect_fraud(flow)
        receipt = emit_hope_florida_receipt(flow, (detected, details))
        receipts.append(receipt)

        if flow.is_fraud and detected:
            true_positives += 1
        elif flow.is_fraud and not detected:
            false_negatives += 1
        elif not flow.is_fraud and detected:
            false_positives += 1
        else:
            true_negatives += 1

    total_fraud = true_positives + false_negatives
    detection_rate = true_positives / total_fraud if total_fraud > 0 else 1.0

    return {
        "n_settlements": n_settlements,
        "fraud_rate": fraud_rate,
        "true_positives": true_positives,
        "false_positives": false_positives,
        "true_negatives": true_negatives,
        "false_negatives": false_negatives,
        "detection_rate": detection_rate,
        "precision": true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0.0,
        "receipts": receipts,
    }
