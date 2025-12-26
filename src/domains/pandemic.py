"""Pandemic Fraud Aggregation.

Aggregates individual PPP/unemployment fraud patterns.

The Florida Pandemic Pattern:
- No systemic DEO fraud equivalent to Minnesota's Feeding Our Future
- Individual/small-scale COVID relief misuse
- Federal prosecutions ongoing
- Focus on aggregation to detect emerging patterns
"""

import random
from dataclasses import dataclass, field
from typing import Any

from ..core import TENANT_ID, emit_receipt


@dataclass
class PPPApplication:
    """Represents a PPP loan application."""

    case_id: str
    business_name: str
    claimed_employees: int
    loan_amount: float
    actual_employees: int | None = None
    business_exists: bool = True
    luxury_purchases: list = field(default_factory=list)
    is_fraud: bool = False
    fraud_type: str | None = None
    network_id: str | None = None


@dataclass
class UnemploymentClaim:
    """Represents an unemployment claim."""

    case_id: str
    claimant_id: str
    weekly_benefit: float
    claim_weeks: int
    employed_during_claim: bool = False
    is_fraud: bool = False
    fraud_type: str | None = None
    network_id: str | None = None


def generate_ppp_applications(
    n_apps: int, fraud_rate: float, seed: int
) -> list[PPPApplication]:
    """Generate synthetic PPP loan applications."""
    rng = random.Random(seed)

    applications = []
    for i in range(n_apps):
        claimed_employees = rng.randint(1, 100)
        app = PPPApplication(
            case_id=f"PPP-{seed}-{i:05d}",
            business_name=f"Business-{rng.randint(1000, 9999)}",
            claimed_employees=claimed_employees,
            loan_amount=claimed_employees * 2500 * rng.uniform(0.8, 1.2),
            actual_employees=claimed_employees,
            business_exists=True,
        )
        applications.append(app)

    return inject_ppp_fraud(applications, fraud_rate, rng)


def inject_ppp_fraud(
    applications: list[PPPApplication], fraud_rate: float, rng: random.Random | None = None
) -> list[PPPApplication]:
    """Inject fraud patterns into PPP applications."""
    if rng is None:
        rng = random.Random()

    fraud_types = ["fake_business", "inflated_employees", "luxury_purchase", "duplicate"]

    # Create some fraud networks
    n_networks = max(1, int(len(applications) * fraud_rate * 0.3))
    networks = [f"NETWORK-{i:03d}" for i in range(n_networks)]

    for app in applications:
        if rng.random() < fraud_rate:
            app.is_fraud = True
            app.fraud_type = rng.choice(fraud_types)

            if app.fraud_type == "fake_business":
                app.business_exists = False
                app.actual_employees = 0
            elif app.fraud_type == "inflated_employees":
                app.actual_employees = max(1, app.claimed_employees // rng.randint(2, 5))
            elif app.fraud_type == "luxury_purchase":
                app.luxury_purchases = rng.sample(
                    ["luxury_car", "jewelry", "real_estate", "boat", "designer_goods"],
                    rng.randint(1, 3),
                )
            elif app.fraud_type == "duplicate":
                app.network_id = rng.choice(networks)

    return applications


def generate_unemployment_claims(
    n_claims: int, fraud_rate: float, seed: int
) -> list[UnemploymentClaim]:
    """Generate synthetic unemployment claims."""
    rng = random.Random(seed)

    claims = []
    for i in range(n_claims):
        claim = UnemploymentClaim(
            case_id=f"UI-{seed}-{i:05d}",
            claimant_id=f"CLAIMANT-{rng.randint(10000, 99999)}",
            weekly_benefit=rng.uniform(200, 600),
            claim_weeks=rng.randint(4, 52),
            employed_during_claim=False,
        )
        claims.append(claim)

    return inject_unemployment_fraud(claims, fraud_rate, rng)


def inject_unemployment_fraud(
    claims: list[UnemploymentClaim], fraud_rate: float, rng: random.Random | None = None
) -> list[UnemploymentClaim]:
    """Inject fraud patterns into unemployment claims."""
    if rng is None:
        rng = random.Random()

    fraud_types = ["employed_while_claiming", "identity_theft", "duplicate"]

    n_networks = max(1, int(len(claims) * fraud_rate * 0.2))
    networks = [f"UI-NETWORK-{i:03d}" for i in range(n_networks)]

    for claim in claims:
        if rng.random() < fraud_rate:
            claim.is_fraud = True
            claim.fraud_type = rng.choice(fraud_types)

            if claim.fraud_type == "employed_while_claiming":
                claim.employed_during_claim = True
            elif claim.fraud_type == "duplicate":
                claim.network_id = rng.choice(networks)

    return claims


def detect_individual_fraud(app: PPPApplication | UnemploymentClaim) -> dict:
    """Check for fraud indicators in individual application."""
    if isinstance(app, PPPApplication):
        return detect_ppp_fraud(app)
    else:
        return detect_unemployment_fraud(app)


def detect_ppp_fraud(app: PPPApplication) -> dict:
    """Detect fraud in PPP application."""
    indicators = {
        "business_not_exists": not app.business_exists,
        "employee_inflation": (
            app.actual_employees is not None
            and app.claimed_employees > app.actual_employees * 1.5
        ),
        "luxury_purchases": len(app.luxury_purchases) > 0,
        "network_member": app.network_id is not None,
    }

    score = sum(1 for v in indicators.values() if v) / len(indicators)

    return {
        "fraud_suspected": score > 0.25,
        "indicators": indicators,
        "score": score,
        "fraud_type": "ppp",
    }


def detect_unemployment_fraud(claim: UnemploymentClaim) -> dict:
    """Detect fraud in unemployment claim."""
    indicators = {
        "employed_during_claim": claim.employed_during_claim,
        "network_member": claim.network_id is not None,
        "excessive_weeks": claim.claim_weeks > 40,
    }

    score = sum(1 for v in indicators.values() if v) / len(indicators)

    return {
        "fraud_suspected": score > 0.25,
        "indicators": indicators,
        "score": score,
        "fraud_type": "unemployment",
    }


def aggregate_fraud_patterns(frauds: list[dict]) -> dict:
    """Roll up individual fraud to detect systemic patterns."""
    if not frauds:
        return {"systemic": False, "patterns": [], "network_count": 0}

    # Count fraud types
    fraud_types: dict[str, int] = {}
    networks: set = set()

    for f in frauds:
        if f.get("fraud_suspected"):
            ftype = f.get("fraud_type", "unknown")
            fraud_types[ftype] = fraud_types.get(ftype, 0) + 1

            indicators = f.get("indicators", {})
            if indicators.get("network_member"):
                networks.add(f.get("network_id", "unknown"))

    # Detect systemic patterns
    total_frauds = sum(fraud_types.values())
    systemic = total_frauds > len(frauds) * 0.1 or len(networks) > 3

    return {
        "systemic": systemic,
        "patterns": list(fraud_types.keys()),
        "fraud_counts": fraud_types,
        "network_count": len(networks),
        "total_suspected": total_frauds,
    }


def check_luxury_purchase_correlation(recipient: str, purchases: list) -> bool:
    """Flag luxury purchases inconsistent with claimed need."""
    luxury_items = {"luxury_car", "jewelry", "real_estate", "boat", "designer_goods"}
    return bool(set(purchases) & luxury_items)


def emit_pandemic_receipt(
    case: PPPApplication | UnemploymentClaim, detection_result: dict
) -> dict:
    """Emit receipt for pandemic fraud detection."""
    program = "PPP" if isinstance(case, PPPApplication) else "unemployment"

    return emit_receipt(
        "pandemic_fraud",
        {
            "tenant_id": TENANT_ID,
            "case_id": case.case_id,
            "program": program,
            "amount": (
                case.loan_amount
                if isinstance(case, PPPApplication)
                else case.weekly_benefit * case.claim_weeks
            ),
            "fraud_type": case.fraud_type,
            "network_id": case.network_id,
            "fraud_confirmed": detection_result["fraud_suspected"],
        },
    )


def run_detection(
    n_ppp: int = 500,
    n_unemployment: int = 500,
    fraud_rate: float = 0.05,
    seed: int = 42,
) -> dict:
    """Run pandemic fraud detection."""
    ppp_apps = generate_ppp_applications(n_ppp, fraud_rate, seed)
    ui_claims = generate_unemployment_claims(n_unemployment, fraud_rate, seed)

    all_results = []
    receipts = []

    # Process PPP
    ppp_tp, ppp_fp, ppp_tn, ppp_fn = 0, 0, 0, 0
    for app in ppp_apps:
        result = detect_ppp_fraud(app)
        receipt = emit_pandemic_receipt(app, result)
        receipts.append(receipt)
        all_results.append(result)

        if app.is_fraud and result["fraud_suspected"]:
            ppp_tp += 1
        elif app.is_fraud and not result["fraud_suspected"]:
            ppp_fn += 1
        elif not app.is_fraud and result["fraud_suspected"]:
            ppp_fp += 1
        else:
            ppp_tn += 1

    # Process unemployment
    ui_tp, ui_fp, ui_tn, ui_fn = 0, 0, 0, 0
    for claim in ui_claims:
        result = detect_unemployment_fraud(claim)
        receipt = emit_pandemic_receipt(claim, result)
        receipts.append(receipt)
        all_results.append(result)

        if claim.is_fraud and result["fraud_suspected"]:
            ui_tp += 1
        elif claim.is_fraud and not result["fraud_suspected"]:
            ui_fn += 1
        elif not claim.is_fraud and result["fraud_suspected"]:
            ui_fp += 1
        else:
            ui_tn += 1

    # Aggregate patterns
    aggregation = aggregate_fraud_patterns(all_results)

    total_tp = ppp_tp + ui_tp
    total_fn = ppp_fn + ui_fn
    total_fraud = total_tp + total_fn
    detection_rate = total_tp / total_fraud if total_fraud > 0 else 1.0

    return {
        "n_ppp": n_ppp,
        "n_unemployment": n_unemployment,
        "fraud_rate": fraud_rate,
        "ppp_detection": {
            "tp": ppp_tp,
            "fp": ppp_fp,
            "tn": ppp_tn,
            "fn": ppp_fn,
        },
        "unemployment_detection": {
            "tp": ui_tp,
            "fp": ui_fp,
            "tn": ui_tn,
            "fn": ui_fn,
        },
        "detection_rate": detection_rate,
        "aggregation": aggregation,
        "receipts": receipts,
    }
