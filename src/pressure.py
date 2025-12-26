"""NEURON Alpha Political Pressure Resilience.

Pattern from ClaimProof/WaProof: Detection stability under disruption.

Pressure Model (Florida-specific):
- Investigator reassignment (Hope Florida House probe halted)
- Criminal investigation delays (May 2025 inquiry)
- Non-cooperation with subpoenas
- Media counterattack ("baseless smears" framing)
"""

import random
from dataclasses import dataclass, field
from typing import Any

from .core import TENANT_ID, emit_receipt


@dataclass
class PressureState:
    """Current pressure state of the system."""

    pressure_level: float = 0.0  # 0-1, higher = more disruption
    probes_halted: int = 0
    subpoenas_ignored: int = 0
    investigators_reassigned: int = 0
    detection_capacity: float = 1.0  # Remaining detection capability


@dataclass
class ProbeStatus:
    """Status of an investigation probe."""

    probe_id: str
    target: str
    original_capacity: float
    current_capacity: float
    is_halted: bool = False
    halt_reason: str | None = None


def simulate_audit_disruption(
    disruption_level: float, n_investigators: int = 10, seed: int = 42
) -> dict:
    """Model investigator reassignment, timeline extensions."""
    rng = random.Random(seed)

    reassigned = 0
    for _ in range(n_investigators):
        if rng.random() < disruption_level * 0.8:
            reassigned += 1

    # Timeline extensions proportional to disruption
    timeline_extension_days = int(disruption_level * 180)  # Up to 6 months delay

    # Detection capacity reduction
    remaining_capacity = max(0.1, 1.0 - (reassigned / n_investigators) * 0.8)

    return {
        "disruption_level": disruption_level,
        "investigators_original": n_investigators,
        "investigators_reassigned": reassigned,
        "investigators_remaining": n_investigators - reassigned,
        "timeline_extension_days": timeline_extension_days,
        "remaining_detection_capacity": remaining_capacity,
    }


def simulate_probe_halt(probe_id: str, pressure: float, seed: int = 42) -> dict:
    """Model House probe halt scenario."""
    rng = random.Random(seed)

    # Probability of halt increases with pressure
    halt_probability = pressure * 0.9

    is_halted = rng.random() < halt_probability

    halt_reasons = [
        "non_cooperation",
        "executive_privilege_claim",
        "legal_challenge",
        "investigator_reassignment",
        "resource_reallocation",
    ]

    probe = ProbeStatus(
        probe_id=probe_id,
        target="Hope Florida Foundation",
        original_capacity=1.0,
        current_capacity=0.0 if is_halted else max(0.2, 1.0 - pressure * 0.5),
        is_halted=is_halted,
        halt_reason=rng.choice(halt_reasons) if is_halted else None,
    )

    return {
        "probe": probe,
        "is_halted": is_halted,
        "halt_reason": probe.halt_reason,
        "remaining_capacity": probe.current_capacity,
    }


def simulate_non_cooperation(
    subpoenas: list[str], pressure: float, seed: int = 42
) -> list[dict]:
    """Model which subpoenas get ignored."""
    rng = random.Random(seed)

    results = []
    for subpoena_id in subpoenas:
        # Higher pressure = more likely to ignore
        ignore_probability = pressure * 0.85

        ignored = rng.random() < ignore_probability

        results.append({
            "subpoena_id": subpoena_id,
            "ignored": ignored,
            "delay_days": rng.randint(30, 180) if not ignored else None,
        })

    return results


def compute_alpha(detection_baseline: float, detection_under_pressure: float) -> float:
    """NEURON alpha = detection stability under disruption.

    alpha = detection_under_pressure / detection_baseline

    alpha >= 0.7 at 50% disruption means the system catches 70% of what
    it would catch in peaceful conditions.
    """
    if detection_baseline <= 0:
        return 0.0

    alpha = detection_under_pressure / detection_baseline
    return min(1.0, max(0.0, alpha))


def simulate_pressure_scenario(
    system_state: dict, pressure_level: float, seed: int = 42
) -> dict:
    """Run full pressure test scenario."""
    rng = random.Random(seed)

    # Initial detection capacity
    baseline_detection = system_state.get("detection_rate", 0.92)

    # Simulate disruptions
    audit_disruption = simulate_audit_disruption(pressure_level, seed=seed)

    # Simulate probe halts
    probes = system_state.get("active_probes", ["PROBE-001", "PROBE-002", "PROBE-003"])
    probe_results = []
    probes_halted = 0
    for probe_id in probes:
        result = simulate_probe_halt(probe_id, pressure_level, seed=seed + hash(probe_id))
        probe_results.append(result)
        if result["is_halted"]:
            probes_halted += 1

    # Simulate subpoena non-cooperation
    subpoenas = system_state.get(
        "pending_subpoenas",
        [f"SUB-{i:03d}" for i in range(5)],
    )
    subpoena_results = simulate_non_cooperation(subpoenas, pressure_level, seed)
    subpoenas_ignored = sum(1 for s in subpoena_results if s["ignored"])

    # Calculate detection under pressure
    capacity_reduction = (
        audit_disruption["remaining_detection_capacity"]
        * (1 - probes_halted / max(len(probes), 1) * 0.3)
        * (1 - subpoenas_ignored / max(len(subpoenas), 1) * 0.2)
    )

    detection_under_pressure = baseline_detection * capacity_reduction

    # Compute NEURON alpha
    alpha = compute_alpha(baseline_detection, detection_under_pressure)

    pressure_state = PressureState(
        pressure_level=pressure_level,
        probes_halted=probes_halted,
        subpoenas_ignored=subpoenas_ignored,
        investigators_reassigned=audit_disruption["investigators_reassigned"],
        detection_capacity=capacity_reduction,
    )

    return {
        "pressure_level": pressure_level,
        "baseline_detection": baseline_detection,
        "detection_under_pressure": detection_under_pressure,
        "alpha": alpha,
        "pressure_state": pressure_state,
        "audit_disruption": audit_disruption,
        "probe_results": probe_results,
        "subpoena_results": subpoena_results,
    }


def run_pressure_sweep(
    system_state: dict | None = None,
    pressure_levels: list | None = None,
    seed: int = 42,
) -> dict:
    """Run pressure test across multiple levels."""
    if system_state is None:
        system_state = {
            "detection_rate": 0.92,
            "active_probes": ["HOPE-PROBE-001", "INSURANCE-PROBE-001"],
            "pending_subpoenas": [f"SUB-{i:03d}" for i in range(10)],
        }

    if pressure_levels is None:
        pressure_levels = [0.0, 0.25, 0.50, 0.75, 1.0]

    results = []
    for pressure in pressure_levels:
        result = simulate_pressure_scenario(system_state, pressure, seed)
        results.append(result)

    # Find alpha at 50% and 75% pressure
    alpha_50 = next((r["alpha"] for r in results if r["pressure_level"] == 0.50), None)
    alpha_75 = next((r["alpha"] for r in results if r["pressure_level"] == 0.75), None)

    # Check SLO compliance
    slo_50_passed = alpha_50 is not None and alpha_50 >= 0.7
    slo_75_passed = alpha_75 is not None and alpha_75 >= 0.5

    return {
        "pressure_levels": pressure_levels,
        "results": results,
        "alpha_50": alpha_50,
        "alpha_75": alpha_75,
        "slo_50_passed": slo_50_passed,
        "slo_75_passed": slo_75_passed,
    }


def emit_pressure_receipt(scenario_result: dict) -> dict:
    """Emit receipt for pressure test."""
    return emit_receipt(
        "pressure",
        {
            "tenant_id": TENANT_ID,
            "pressure_level": scenario_result["pressure_level"],
            "detection_baseline": scenario_result["baseline_detection"],
            "detection_under_pressure": scenario_result["detection_under_pressure"],
            "alpha": scenario_result["alpha"],
            "probes_halted": scenario_result["pressure_state"].probes_halted,
            "subpoenas_ignored": scenario_result["pressure_state"].subpoenas_ignored,
        },
    )


def run_pressure_analysis(seed: int = 42) -> dict:
    """Run complete pressure analysis."""
    system_state = {
        "detection_rate": 0.92,
        "active_probes": [
            "HOPE-FLORIDA-HOUSE-PROBE",
            "INSURANCE-REFORM-PROBE",
            "SHERIFF-CONTRACTS-PROBE",
        ],
        "pending_subpoenas": [f"SUBPOENA-{i:03d}" for i in range(15)],
    }

    sweep_result = run_pressure_sweep(system_state, seed=seed)

    # Emit receipts for each pressure level
    receipts = []
    for result in sweep_result["results"]:
        receipt = emit_pressure_receipt(result)
        receipts.append(receipt)

    return {
        "sweep_result": sweep_result,
        "alpha_50": sweep_result["alpha_50"],
        "alpha_75": sweep_result["alpha_75"],
        "slo_passed": sweep_result["slo_50_passed"] and sweep_result["slo_75_passed"],
        "receipts": receipts,
    }
