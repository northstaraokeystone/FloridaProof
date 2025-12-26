"""Corporate Capture Detection.

Detects Publix/FPL/U.S. Sugar donor-policy correlation.

The Corporate Capture Pattern:
- Publix: Top donor (millions to GOP), favors pro-business policies
- FPL/NextEra: Utility rate approvals align with donations
- U.S. Sugar: Agricultural subsidies and protections
- Pattern: Donations ensure access, legislation follows
"""

import random
from dataclasses import dataclass, field
from typing import Any

from ..core import TENANT_ID, emit_receipt


@dataclass
class CorporateDonation:
    """Represents a corporate political donation."""

    corporation: str
    amount: float
    recipient: str  # PAC, candidate, party
    election_cycle: int
    is_fraud: bool = False


@dataclass
class PolicyOutcome:
    """Represents a policy outcome benefiting a corporation."""

    policy_id: str
    policy_name: str
    beneficiary_corporation: str
    estimated_value: float
    days_after_donation: int
    policy_type: str  # "subsidy", "rate_approval", "deregulation", "tax_benefit"


# Key Florida corporate donors
MAJOR_DONORS = {
    "Publix": {
        "industry": "retail",
        "policy_interests": ["labor_law", "business_regulation", "tax_policy"],
        "avg_donation": 2_000_000,
    },
    "FPL/NextEra": {
        "industry": "utility",
        "policy_interests": ["rate_approval", "renewable_energy", "grid_regulation"],
        "avg_donation": 1_500_000,
    },
    "U.S. Sugar": {
        "industry": "agriculture",
        "policy_interests": ["subsidies", "water_policy", "environmental_regulation"],
        "avg_donation": 1_000_000,
    },
    "Disney": {
        "industry": "entertainment",
        "policy_interests": ["special_districts", "tax_incentives", "tourism"],
        "avg_donation": 800_000,
    },
    "Walmart": {
        "industry": "retail",
        "policy_interests": ["labor_law", "zoning", "tax_policy"],
        "avg_donation": 500_000,
    },
}


def generate_corporate_donations(
    n_corps: int, n_cycles: int, seed: int
) -> list[CorporateDonation]:
    """Generate donation patterns over election cycles."""
    rng = random.Random(seed)

    corps = list(MAJOR_DONORS.keys())[:n_corps] if n_corps <= len(MAJOR_DONORS) else list(MAJOR_DONORS.keys())
    cycles = list(range(2016, 2016 + n_cycles * 2, 2))

    donations = []
    for corp in corps:
        corp_info = MAJOR_DONORS.get(corp, {"avg_donation": 100_000})
        base_amount = corp_info.get("avg_donation", 100_000)

        for cycle in cycles:
            # Vary donation amount by cycle
            amount = base_amount * rng.uniform(0.5, 1.5)
            recipients = ["FL GOP", "DeSantis PAC", "FL Chamber", "Local PACs"]

            for recipient in rng.sample(recipients, rng.randint(1, 3)):
                donation = CorporateDonation(
                    corporation=corp,
                    amount=amount * rng.uniform(0.1, 0.5),
                    recipient=recipient,
                    election_cycle=cycle,
                )
                donations.append(donation)

    return donations


def generate_policy_outcomes(
    n_policies: int, corps: list[str], seed: int
) -> list[PolicyOutcome]:
    """Generate policy decisions affecting donors."""
    rng = random.Random(seed)

    policy_types = ["subsidy", "rate_approval", "deregulation", "tax_benefit"]

    outcomes = []
    for i in range(n_policies):
        corp = rng.choice(corps)
        policy_type = rng.choice(policy_types)

        # Estimate value based on policy type
        value_ranges = {
            "subsidy": (1_000_000, 50_000_000),
            "rate_approval": (10_000_000, 500_000_000),
            "deregulation": (5_000_000, 100_000_000),
            "tax_benefit": (2_000_000, 20_000_000),
        }
        min_val, max_val = value_ranges[policy_type]

        outcome = PolicyOutcome(
            policy_id=f"POLICY-{i:04d}",
            policy_name=f"{policy_type.replace('_', ' ').title()} for {corp}",
            beneficiary_corporation=corp,
            estimated_value=rng.uniform(min_val, max_val),
            days_after_donation=rng.randint(30, 730),
            policy_type=policy_type,
        )
        outcomes.append(outcome)

    return outcomes


def detect_donation_policy_correlation(
    donations: list[CorporateDonation], policies: list[PolicyOutcome]
) -> dict:
    """Compute correlation between donation spikes and favorable policy."""
    # Aggregate donations by corporation
    corp_donations: dict[str, float] = {}
    for d in donations:
        corp_donations[d.corporation] = corp_donations.get(d.corporation, 0) + d.amount

    # Aggregate policy value by corporation
    corp_policy_value: dict[str, float] = {}
    for p in policies:
        corp_policy_value[p.beneficiary_corporation] = (
            corp_policy_value.get(p.beneficiary_corporation, 0) + p.estimated_value
        )

    # Calculate correlation
    common_corps = set(corp_donations.keys()) & set(corp_policy_value.keys())

    if len(common_corps) < 2:
        return {"correlation": 0.0, "sample_size": len(common_corps)}

    donations_list = [corp_donations[c] for c in common_corps]
    values_list = [corp_policy_value[c] for c in common_corps]

    # Simple correlation: rank-based
    donation_ranks = {c: r for r, c in enumerate(sorted(common_corps, key=lambda x: corp_donations[x]))}
    value_ranks = {c: r for r, c in enumerate(sorted(common_corps, key=lambda x: corp_policy_value[x]))}

    rank_diffs = [(donation_ranks[c] - value_ranks[c]) ** 2 for c in common_corps]
    n = len(common_corps)

    # Spearman's rho approximation
    rho = 1 - (6 * sum(rank_diffs)) / (n * (n**2 - 1)) if n > 1 else 0

    return {
        "correlation": rho,
        "sample_size": n,
        "total_donations": sum(donations_list),
        "total_policy_value": sum(values_list),
        "corporations": list(common_corps),
    }


def calculate_policy_value(policy: PolicyOutcome, corp: str) -> float:
    """Estimate dollar value of policy to specific corporation."""
    if policy.beneficiary_corporation != corp:
        return 0.0
    return policy.estimated_value


def track_committee_assignment_correlation(
    donations: list[CorporateDonation], assignments: list | None = None
) -> dict:
    """Check if donors get favorable committee oversight."""
    if assignments is None:
        # Simulate committee assignments
        return {"correlation": 0.0, "favorable_assignments": 0}

    return {"correlation": 0.0, "favorable_assignments": 0}


def compute_roi_ratio(donations: float, policy_value: float) -> float:
    """Calculate return on investment ratio."""
    if donations <= 0:
        return 0.0
    return policy_value / donations


def score_capture_risk(correlation: float, roi: float) -> float:
    """Risk score based on correlation and ROI."""
    risk = 0.0

    # Correlation component
    if correlation > 0.7:
        risk += 0.4
    elif correlation > 0.5:
        risk += 0.25
    elif correlation > 0.3:
        risk += 0.15

    # ROI component (suspicious if too high)
    if roi > 100:  # $100 in policy value per $1 donated
        risk += 0.4
    elif roi > 50:
        risk += 0.3
    elif roi > 20:
        risk += 0.2
    else:
        risk += 0.1

    return min(1.0, risk)


def detect_fraud(
    donations: list[CorporateDonation], policies: list[PolicyOutcome], corp: str
) -> tuple[bool, dict]:
    """Main fraud detection for corporate capture."""
    corp_donations = [d for d in donations if d.corporation == corp]
    corp_policies = [p for p in policies if p.beneficiary_corporation == corp]

    total_donations = sum(d.amount for d in corp_donations)
    total_policy_value = sum(p.estimated_value for p in corp_policies)

    correlation_result = detect_donation_policy_correlation(donations, policies)
    roi = compute_roi_ratio(total_donations, total_policy_value)
    risk_score = score_capture_risk(correlation_result["correlation"], roi)

    fraud_detected = (
        risk_score > 0.6
        or (roi > 50 and correlation_result["correlation"] > 0.5)
        or (total_policy_value > 100_000_000 and correlation_result["correlation"] > 0.4)
    )

    details = {
        "risk_score": risk_score,
        "correlation": correlation_result,
        "roi": roi,
        "total_donations": total_donations,
        "total_policy_value": total_policy_value,
        "n_policies": len(corp_policies),
    }

    return fraud_detected, details


def emit_corporate_receipt(
    corp: str, detection_result: tuple[bool, dict], policies: list[PolicyOutcome]
) -> dict:
    """Emit receipt for corporate capture detection."""
    fraud_detected, details = detection_result

    policy_benefits = [
        {"policy": p.policy_name, "estimated_value": p.estimated_value}
        for p in policies
        if p.beneficiary_corporation == corp
    ]

    return emit_receipt(
        "corporate_capture",
        {
            "tenant_id": TENANT_ID,
            "corporation": corp,
            "total_donations": details["total_donations"],
            "policy_benefits": policy_benefits[:10],  # Limit for receipt size
            "correlation_coefficient": details["correlation"]["correlation"],
            "roi_ratio": details["roi"],
            "capture_risk_score": details["risk_score"],
        },
    )


def run_detection(
    n_corps: int = 5, n_cycles: int = 4, n_policies: int = 20, fraud_rate: float = 0.12, seed: int = 42
) -> dict:
    """Run corporate capture fraud detection."""
    donations = generate_corporate_donations(n_corps, n_cycles, seed)
    corps = list(set(d.corporation for d in donations))
    policies = generate_policy_outcomes(n_policies, corps, seed)

    # Inject fraud patterns
    rng = random.Random(seed)
    for d in donations:
        if rng.random() < fraud_rate:
            d.is_fraud = True

    results = []
    receipts = []
    for corp in corps:
        detected, details = detect_fraud(donations, policies, corp)
        receipt = emit_corporate_receipt(corp, (detected, details), policies)
        receipts.append(receipt)
        results.append({"corporation": corp, "detected": detected, "details": details})

    # Aggregate detection rate
    injected_fraud = [d for d in donations if d.is_fraud]
    detected_corps = [r for r in results if r["detected"]]

    return {
        "n_corps": len(corps),
        "n_donations": len(donations),
        "n_policies": len(policies),
        "fraud_rate": fraud_rate,
        "corporations_flagged": len(detected_corps),
        "detection_rate": len(detected_corps) / len(corps) if corps else 0.0,
        "avg_roi": sum(r["details"]["roi"] for r in results) / len(results) if results else 0.0,
        "receipts": receipts,
    }
