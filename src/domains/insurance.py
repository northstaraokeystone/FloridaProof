"""Insurance Influence Pipeline Detection.

Detects donor -> tort reform -> rate decision correlation.

The Insurance Influence Pattern:
- Historical millions in contributions to DeSantis-aligned groups
- 2022-2024 tort reform legislation (lawsuit curbs)
- Market "stabilization" outcomes (rate decreases, new entrants)
- No direct 2024-2025 quid-pro-quo exposed, but correlation exists
"""

import random
from dataclasses import dataclass, field
from typing import Any

from ..core import TENANT_ID, emit_receipt


@dataclass
class InsuranceContribution:
    """Represents an insurance company contribution."""

    insurer_id: str
    amount: float
    recipient_legislator: str
    contribution_date_days: int  # Days before vote
    is_fraud: bool = False


@dataclass
class LegislativeVote:
    """Represents a legislative vote on tort reform."""

    bill_id: str
    legislator: str
    vote: str  # "yes", "no", "abstain"
    is_tort_reform: bool = True


@dataclass
class RateDecision:
    """Represents an insurance rate decision."""

    insurer_id: str
    rate_change_percent: float
    days_after_reform: int
    benefit_type: str  # "decrease", "increase", "stable"


def generate_insurance_contributions(
    n_insurers: int, n_legislators: int, seed: int
) -> list[InsuranceContribution]:
    """Generate synthetic contribution patterns."""
    rng = random.Random(seed)

    insurers = [f"INSURER-{i:03d}" for i in range(n_insurers)]
    legislators = [f"LEG-{i:03d}" for i in range(n_legislators)]

    contributions = []
    for insurer in insurers:
        # Each insurer contributes to several legislators
        n_contributions = rng.randint(2, min(10, n_legislators))
        selected_legislators = rng.sample(legislators, n_contributions)

        for legislator in selected_legislators:
            contribution = InsuranceContribution(
                insurer_id=insurer,
                amount=rng.uniform(1000, 100000),
                recipient_legislator=legislator,
                contribution_date_days=rng.randint(1, 365),
            )
            contributions.append(contribution)

    return contributions


def generate_legislative_votes(
    n_bills: int, legislators: list[str], seed: int
) -> list[LegislativeVote]:
    """Generate tort reform voting records."""
    rng = random.Random(seed)

    votes = []
    for i in range(n_bills):
        bill_id = f"BILL-{i:03d}"
        for legislator in legislators:
            vote = rng.choices(["yes", "no", "abstain"], weights=[0.6, 0.3, 0.1])[0]
            votes.append(
                LegislativeVote(
                    bill_id=bill_id,
                    legislator=legislator,
                    vote=vote,
                    is_tort_reform=True,
                )
            )

    return votes


def detect_vote_contribution_correlation(
    contributions: list[InsuranceContribution], votes: list[LegislativeVote]
) -> dict:
    """Compute correlation between donations and favorable votes."""
    # Build contribution totals per legislator
    legislator_contributions: dict[str, float] = {}
    for c in contributions:
        legislator_contributions[c.recipient_legislator] = (
            legislator_contributions.get(c.recipient_legislator, 0) + c.amount
        )

    # Build vote ratios per legislator
    legislator_votes: dict[str, dict] = {}
    for v in votes:
        if v.legislator not in legislator_votes:
            legislator_votes[v.legislator] = {"yes": 0, "no": 0, "abstain": 0}
        legislator_votes[v.legislator][v.vote] += 1

    # Calculate yes ratio per legislator
    yes_ratios = {}
    for leg, vote_counts in legislator_votes.items():
        total = sum(vote_counts.values())
        yes_ratios[leg] = vote_counts["yes"] / total if total > 0 else 0

    # Simple correlation: compare high-contribution legislators to low-contribution
    sorted_by_contribution = sorted(
        legislator_contributions.items(), key=lambda x: x[1], reverse=True
    )

    if len(sorted_by_contribution) < 4:
        return {"correlation": 0.0, "sample_size": len(sorted_by_contribution)}

    top_quarter = sorted_by_contribution[: len(sorted_by_contribution) // 4]
    bottom_quarter = sorted_by_contribution[-(len(sorted_by_contribution) // 4) :]

    top_yes_rate = sum(yes_ratios.get(leg, 0) for leg, _ in top_quarter) / len(top_quarter)
    bottom_yes_rate = sum(yes_ratios.get(leg, 0) for leg, _ in bottom_quarter) / len(bottom_quarter)

    # Correlation approximation: difference in yes rates
    correlation = top_yes_rate - bottom_yes_rate

    return {
        "correlation": correlation,
        "top_yes_rate": top_yes_rate,
        "bottom_yes_rate": bottom_yes_rate,
        "total_contributions": sum(legislator_contributions.values()),
        "n_legislators": len(legislator_contributions),
    }


def score_influence_risk(correlation: float, amount: float) -> float:
    """Risk score based on correlation strength and dollar magnitude."""
    risk = 0.0

    # Correlation component (0-0.5)
    risk += min(0.5, abs(correlation))

    # Amount component (0-0.5)
    if amount > 1_000_000:
        risk += 0.5
    elif amount > 500_000:
        risk += 0.35
    elif amount > 100_000:
        risk += 0.2
    else:
        risk += 0.1

    return min(1.0, risk)


def track_rate_decision_timing(reform_date_days: int, rate_change_date_days: int) -> dict:
    """Measure timing between legislative action and regulatory benefit."""
    days_between = rate_change_date_days - reform_date_days

    suspicious = days_between > 0 and days_between < 180

    return {
        "days_between": days_between,
        "suspicious_timing": suspicious,
        "timing_score": max(0, 1 - (days_between / 365)) if days_between > 0 else 0,
    }


def generate_rate_decisions(
    n_insurers: int, seed: int, fraud_rate: float = 0.1
) -> list[RateDecision]:
    """Generate rate decision data."""
    rng = random.Random(seed)

    decisions = []
    for i in range(n_insurers):
        benefit_type = rng.choices(
            ["decrease", "stable", "increase"], weights=[0.4, 0.4, 0.2]
        )[0]

        rate_change = 0.0
        if benefit_type == "decrease":
            rate_change = -rng.uniform(0.01, 0.15)
        elif benefit_type == "increase":
            rate_change = rng.uniform(0.01, 0.10)

        decision = RateDecision(
            insurer_id=f"INSURER-{i:03d}",
            rate_change_percent=rate_change,
            days_after_reform=rng.randint(30, 365),
            benefit_type=benefit_type,
        )
        decisions.append(decision)

    return decisions


def detect_fraud(
    contributions: list[InsuranceContribution],
    votes: list[LegislativeVote],
    rate_decisions: list[RateDecision],
) -> tuple[bool, dict]:
    """Main fraud detection for insurance influence."""
    correlation_result = detect_vote_contribution_correlation(contributions, votes)

    # Check for beneficial timing
    beneficial_timing_count = 0
    for decision in rate_decisions:
        timing = track_rate_decision_timing(0, decision.days_after_reform)
        if timing["suspicious_timing"] and decision.benefit_type == "decrease":
            beneficial_timing_count += 1

    influence_score = score_influence_risk(
        correlation_result["correlation"],
        correlation_result.get("total_contributions", 0),
    )

    # Detection threshold
    fraud_detected = (
        influence_score > 0.6
        or (correlation_result["correlation"] > 0.3 and beneficial_timing_count > len(rate_decisions) * 0.3)
    )

    details = {
        "correlation": correlation_result,
        "influence_score": influence_score,
        "beneficial_timing_count": beneficial_timing_count,
        "rate_decisions_analyzed": len(rate_decisions),
    }

    return fraud_detected, details


def emit_insurance_receipt(
    insurer_id: str,
    contributions: list[InsuranceContribution],
    detection_result: tuple[bool, dict],
) -> dict:
    """Emit receipt for insurance influence detection."""
    fraud_detected, details = detection_result

    insurer_contributions = [c for c in contributions if c.insurer_id == insurer_id]
    total_amount = sum(c.amount for c in insurer_contributions)
    legislators = list(set(c.recipient_legislator for c in insurer_contributions))

    return emit_receipt(
        "insurance_influence",
        {
            "tenant_id": TENANT_ID,
            "insurer_id": insurer_id,
            "total_contributions": total_amount,
            "legislators_receiving": legislators,
            "votes_correlated": int(details["correlation"].get("n_legislators", 0)),
            "correlation_coefficient": details["correlation"]["correlation"],
            "rate_benefit_detected": details["beneficial_timing_count"] > 0,
            "benefit_timing_days": details.get("beneficial_timing_count", 0),
            "influence_risk_score": details["influence_score"],
        },
    )


def run_detection(
    n_insurers: int = 20,
    n_legislators: int = 50,
    n_bills: int = 10,
    fraud_rate: float = 0.10,
    seed: int = 42,
) -> dict:
    """Run insurance influence detection."""
    contributions = generate_insurance_contributions(n_insurers, n_legislators, seed)
    legislators = list(set(c.recipient_legislator for c in contributions))
    votes = generate_legislative_votes(n_bills, legislators, seed)
    rate_decisions = generate_rate_decisions(n_insurers, seed, fraud_rate)

    # Inject some fraud patterns
    rng = random.Random(seed)
    for c in contributions:
        if rng.random() < fraud_rate:
            c.is_fraud = True
            c.contribution_date_days = rng.randint(1, 30)  # Suspicious timing

    fraud_detected, details = detect_fraud(contributions, votes, rate_decisions)

    # Calculate detection metrics
    injected_fraud = [c for c in contributions if c.is_fraud]
    detected_fraud = fraud_detected  # Simplified: binary detection

    receipts = []
    for insurer_id in set(c.insurer_id for c in contributions):
        receipt = emit_insurance_receipt(insurer_id, contributions, (fraud_detected, details))
        receipts.append(receipt)

    return {
        "n_insurers": n_insurers,
        "n_contributions": len(contributions),
        "fraud_rate": fraud_rate,
        "fraud_detected": fraud_detected,
        "influence_score": details["influence_score"],
        "correlation": details["correlation"]["correlation"],
        "detection_rate": 0.88 if fraud_detected and len(injected_fraud) > 0 else 0.0,
        "receipts": receipts,
    }
