"""Sheriff Corruption Pattern Detection.

Detects no-show contracts and kickback patterns.

The Marceno Pattern (documented):
- Lee County Sheriff Carmine Marceno
- $5,700/month no-show consulting contract to jeweler Ken Romano
- Allegations: part funneled to Marceno's father (luxury car)
- Gambling ties, unreported gifts
- Federal probe closed November 2025 without charges
"""

import random
from dataclasses import dataclass, field
from typing import Any

from ..core import TENANT_ID, emit_receipt


@dataclass
class SheriffContract:
    """Represents a sheriff office contract."""

    contract_id: str
    sheriff_office: str
    contractor: str
    monthly_amount: float
    contract_type: str  # "consulting", "services", "equipment"
    deliverables_documented: bool = True
    personal_tie_detected: bool = False
    family_connection: bool = False
    gambling_flag: bool = False
    is_fraud: bool = False
    fraud_type: str | None = None


@dataclass
class FinancialLink:
    """Represents a financial link for kickback detection."""

    source_entity: str
    target_entity: str
    relationship: str  # "family", "business", "personal"
    amount_transferred: float = 0.0


# Known sheriff offices
SHERIFF_OFFICES = [
    "Lee County",
    "Broward County",
    "Miami-Dade County",
    "Hillsborough County",
    "Orange County",
    "Pinellas County",
    "Palm Beach County",
]


def generate_sheriff_contracts(
    n_contracts: int, fraud_rate: float, seed: int
) -> list[SheriffContract]:
    """Generate synthetic sheriff office contracts."""
    rng = random.Random(seed)

    contracts = []
    for i in range(n_contracts):
        contract = SheriffContract(
            contract_id=f"CONTRACT-{seed}-{i:04d}",
            sheriff_office=rng.choice(SHERIFF_OFFICES),
            contractor=f"CONTRACTOR-{rng.randint(1, 100):03d}",
            monthly_amount=rng.uniform(1000, 50000),
            contract_type=rng.choice(["consulting", "services", "equipment"]),
            deliverables_documented=rng.random() > 0.1,
        )
        contracts.append(contract)

    return inject_contract_fraud(contracts, fraud_rate, rng)


def inject_contract_fraud(
    contracts: list[SheriffContract], fraud_rate: float, rng: random.Random | None = None
) -> list[SheriffContract]:
    """Inject fraud patterns into contracts."""
    if rng is None:
        rng = random.Random()

    fraud_types = ["no_show", "kickback", "inflated", "personal_tie"]

    for contract in contracts:
        if rng.random() < fraud_rate:
            contract.is_fraud = True
            contract.fraud_type = rng.choice(fraud_types)

            if contract.fraud_type == "no_show":
                contract.deliverables_documented = False
                contract.contract_type = "consulting"
                contract.monthly_amount = rng.uniform(5000, 10000)  # Typical no-show range
            elif contract.fraud_type == "kickback":
                contract.family_connection = True
                contract.personal_tie_detected = True
            elif contract.fraud_type == "inflated":
                contract.monthly_amount *= rng.uniform(2.0, 5.0)
            elif contract.fraud_type == "personal_tie":
                contract.personal_tie_detected = True
                contract.gambling_flag = rng.random() > 0.5

    return contracts


def detect_no_show_pattern(contract: SheriffContract) -> dict:
    """Check for missing deliverables, inflated rates."""
    indicators = {
        "no_deliverables": not contract.deliverables_documented,
        "consulting_type": contract.contract_type == "consulting",
        "suspicious_amount": 4000 < contract.monthly_amount < 15000,
    }

    score = sum(1 for v in indicators.values() if v) / len(indicators)

    return {
        "no_show_suspected": score > 0.5,
        "indicators": indicators,
        "score": score,
    }


def detect_kickback_routing(
    contract: SheriffContract, financial_links: list[FinancialLink] | None = None
) -> dict:
    """Check for payments routed to personal associates."""
    if financial_links is None:
        financial_links = []

    # Check contract flags
    family_link = contract.family_connection
    personal_tie = contract.personal_tie_detected

    # Check financial links
    suspicious_links = []
    for link in financial_links:
        if link.source_entity == contract.contractor:
            if link.relationship in ["family", "personal"]:
                suspicious_links.append(link)
                family_link = True

    kickback_score = 0.0
    if family_link:
        kickback_score += 0.4
    if personal_tie:
        kickback_score += 0.3
    if not contract.deliverables_documented:
        kickback_score += 0.2
    if len(suspicious_links) > 0:
        kickback_score += 0.1 * len(suspicious_links)

    return {
        "kickback_suspected": kickback_score > 0.5,
        "family_link": family_link,
        "personal_tie": personal_tie,
        "suspicious_links": len(suspicious_links),
        "score": min(1.0, kickback_score),
    }


def check_gambling_correlation(
    entity: str, gambling_records: dict | None = None
) -> bool:
    """Flag entities with suspicious gambling patterns."""
    if gambling_records is None:
        return False

    return entity in gambling_records and gambling_records[entity].get("suspicious", False)


def compute_market_rate(contract_type: str) -> tuple[float, float]:
    """Return expected market rate range for contract type."""
    rates = {
        "consulting": (2000, 8000),
        "services": (3000, 15000),
        "equipment": (5000, 50000),
    }
    return rates.get(contract_type, (1000, 10000))


def score_contract_risk(contract: SheriffContract) -> float:
    """Calculate overall fraud risk score."""
    risk = 0.0

    # No-show indicators
    no_show = detect_no_show_pattern(contract)
    risk += no_show["score"] * 0.4

    # Kickback indicators
    kickback = detect_kickback_routing(contract)
    risk += kickback["score"] * 0.4

    # Market rate deviation
    min_rate, max_rate = compute_market_rate(contract.contract_type)
    if contract.monthly_amount > max_rate * 1.5:
        risk += 0.15
    elif contract.monthly_amount < min_rate * 0.5:
        risk += 0.05  # Suspiciously low might indicate hidden compensation

    # Gambling flag
    if contract.gambling_flag:
        risk += 0.1

    return min(1.0, risk)


def detect_fraud(contract: SheriffContract) -> tuple[bool, dict]:
    """Main fraud detection for sheriff contracts."""
    risk_score = score_contract_risk(contract)
    no_show = detect_no_show_pattern(contract)
    kickback = detect_kickback_routing(contract)

    fraud_detected = (
        risk_score > 0.55
        or (no_show["no_show_suspected"] and contract.personal_tie_detected)
        or kickback["kickback_suspected"]
    )

    details = {
        "risk_score": risk_score,
        "no_show": no_show,
        "kickback": kickback,
        "gambling_flag": contract.gambling_flag,
    }

    return fraud_detected, details


def emit_sheriff_receipt(contract: SheriffContract, detection_result: tuple[bool, dict]) -> dict:
    """Emit receipt for sheriff contract fraud detection."""
    fraud_detected, details = detection_result

    return emit_receipt(
        "sheriff_contract",
        {
            "tenant_id": TENANT_ID,
            "contract_id": contract.contract_id,
            "sheriff_office": contract.sheriff_office,
            "contractor": contract.contractor,
            "monthly_amount": contract.monthly_amount,
            "deliverables_documented": contract.deliverables_documented,
            "personal_tie_detected": contract.personal_tie_detected,
            "kickback_suspected": details["kickback"]["kickback_suspected"],
            "gambling_flag": contract.gambling_flag,
            "fraud_risk_score": details["risk_score"],
        },
    )


def run_detection(
    n_contracts: int = 500, fraud_rate: float = 0.08, seed: int = 42
) -> dict:
    """Run sheriff contract fraud detection."""
    contracts = generate_sheriff_contracts(n_contracts, fraud_rate, seed)

    true_positives = 0
    false_positives = 0
    true_negatives = 0
    false_negatives = 0
    receipts = []

    for contract in contracts:
        detected, details = detect_fraud(contract)
        receipt = emit_sheriff_receipt(contract, (detected, details))
        receipts.append(receipt)

        if contract.is_fraud and detected:
            true_positives += 1
        elif contract.is_fraud and not detected:
            false_negatives += 1
        elif not contract.is_fraud and detected:
            false_positives += 1
        else:
            true_negatives += 1

    total_fraud = true_positives + false_negatives
    detection_rate = true_positives / total_fraud if total_fraud > 0 else 1.0

    return {
        "n_contracts": n_contracts,
        "fraud_rate": fraud_rate,
        "true_positives": true_positives,
        "false_positives": false_positives,
        "true_negatives": true_negatives,
        "false_negatives": false_negatives,
        "detection_rate": detection_rate,
        "precision": true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0.0,
        "receipts": receipts,
    }
