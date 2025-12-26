"""Self-Spawning Fraud Detection Agents.

Pattern from ClaimProof/WaProof: Autocatalytic fraud hunters.

Watcher Lifecycle:
1. High fraud signal (>0.7) spawns new watcher
2. Watcher emits detection receipts
3. If watcher's detections predict its own future detections (autocatalysis) -> survives
4. If not self-referencing -> dissolves
5. Entropy-governed population cap
"""

import random
from dataclasses import dataclass, field
from typing import Any, Callable

from .core import TENANT_ID, emit_receipt

# Watcher configuration
WATCHER_CONFIG = {
    "spawn_threshold": 0.7,  # Fraud signal threshold to spawn
    "autocatalysis_threshold": 0.5,  # Self-reference threshold to survive
    "max_watchers": 50,  # Population cap
    "dissolution_after_cycles": 100,  # Cycles without autocatalysis -> dissolve
}


@dataclass
class Watcher:
    """A self-spawning fraud detection agent."""

    watcher_id: str
    target_domain: str
    spawn_cycle: int
    detection_history: list = field(default_factory=list)
    self_reference_score: float = 0.0
    is_autocatalytic: bool = False
    cycles_since_autocatalysis: int = 0
    total_detections: int = 0


@dataclass
class FraudEvent:
    """Represents a fraud event for watcher detection."""

    event_id: str
    domain: str
    fraud_signal: float
    timestamp: int
    detected_by: list = field(default_factory=list)


def spawn_watcher(
    fraud_signal: dict, config: dict | None = None, cycle: int = 0
) -> Watcher | None:
    """Create new watcher when signal > threshold."""
    if config is None:
        config = WATCHER_CONFIG

    domain = fraud_signal.get("domain", "unknown")
    signal_strength = fraud_signal.get("signal", 0)

    if signal_strength < config["spawn_threshold"]:
        return None

    watcher_id = f"WATCHER-{domain[:3].upper()}-{cycle:05d}"

    watcher = Watcher(
        watcher_id=watcher_id,
        target_domain=domain,
        spawn_cycle=cycle,
    )

    # Emit spawn receipt
    emit_watcher_receipt(watcher, "spawn")

    return watcher


def watcher_detect(watcher: Watcher, events: list[FraudEvent]) -> list[dict]:
    """Run watcher detection on event stream."""
    detections = []

    for event in events:
        # Watcher only detects in its target domain
        if event.domain != watcher.target_domain:
            continue

        # Detection probability based on signal strength
        detection_prob = min(0.95, event.fraud_signal * 1.2)

        # Simulate detection (deterministic based on event)
        event_hash = hash(event.event_id + watcher.watcher_id)
        if (event_hash % 100) / 100 < detection_prob:
            detection = {
                "watcher_id": watcher.watcher_id,
                "event_id": event.event_id,
                "domain": event.domain,
                "signal": event.fraud_signal,
                "timestamp": event.timestamp,
            }
            detections.append(detection)
            event.detected_by.append(watcher.watcher_id)
            watcher.total_detections += 1
            watcher.detection_history.append(detection)

    return detections


def check_autocatalysis(watcher: Watcher, receipts: list[dict]) -> bool:
    """Does watcher predict its own future detections?

    Autocatalysis: watcher's detections create conditions for more detections.
    Measured by whether detection patterns are self-reinforcing.
    """
    if len(watcher.detection_history) < 3:
        return False

    # Calculate self-reference score
    # High score if watcher's detections cluster in time/domain
    recent = watcher.detection_history[-10:]

    if len(recent) < 2:
        return False

    # Check temporal clustering (detections close together indicate pattern)
    timestamps = [d["timestamp"] for d in recent]
    if len(timestamps) > 1:
        gaps = [timestamps[i + 1] - timestamps[i] for i in range(len(timestamps) - 1)]
        avg_gap = sum(gaps) / len(gaps) if gaps else float("inf")
        temporal_score = max(0, 1 - (avg_gap / 10))  # Closer gaps = higher score
    else:
        temporal_score = 0

    # Check signal strength trend (increasing signals = pattern emergence)
    signals = [d["signal"] for d in recent]
    if len(signals) > 1:
        trend = (signals[-1] - signals[0]) / len(signals)
        trend_score = max(0, min(1, trend + 0.5))
    else:
        trend_score = 0.5

    # Combined self-reference score
    watcher.self_reference_score = (temporal_score + trend_score) / 2

    # Check against threshold
    config = WATCHER_CONFIG
    if watcher.self_reference_score >= config["autocatalysis_threshold"]:
        watcher.is_autocatalytic = True
        watcher.cycles_since_autocatalysis = 0
        return True

    watcher.cycles_since_autocatalysis += 1
    return False


def dissolve_watcher(watcher: Watcher) -> dict:
    """Clean up non-autocatalytic watcher."""
    receipt = emit_watcher_receipt(watcher, "dissolve")

    return {
        "watcher_id": watcher.watcher_id,
        "reason": "non_autocatalytic",
        "total_detections": watcher.total_detections,
        "final_score": watcher.self_reference_score,
        "receipt": receipt,
    }


def govern_population(watchers: list[Watcher], entropy_budget: float) -> list[Watcher]:
    """Enforce population cap via entropy gradient."""
    config = WATCHER_CONFIG
    max_pop = int(config["max_watchers"] * entropy_budget)

    if len(watchers) <= max_pop:
        return watchers

    # Sort by autocatalysis score (keep highest)
    sorted_watchers = sorted(
        watchers, key=lambda w: (w.is_autocatalytic, w.self_reference_score), reverse=True
    )

    # Keep top watchers, dissolve rest
    survivors = sorted_watchers[:max_pop]
    dissolved = sorted_watchers[max_pop:]

    for watcher in dissolved:
        dissolve_watcher(watcher)

    return survivors


def emit_watcher_receipt(watcher: Watcher, event: str) -> dict:
    """Emit receipt for watcher event."""
    return emit_receipt(
        "watcher",
        {
            "tenant_id": TENANT_ID,
            "watcher_id": watcher.watcher_id,
            "event": event,
            "target_domain": watcher.target_domain,
            "detections_this_cycle": len(
                [d for d in watcher.detection_history[-10:]]
            ) if watcher.detection_history else 0,
            "self_reference_score": watcher.self_reference_score,
            "is_autocatalytic": watcher.is_autocatalytic,
        },
    )


def run_watcher_cycle(
    watchers: list[Watcher],
    events: list[FraudEvent],
    cycle: int,
    config: dict | None = None,
) -> dict:
    """Run one cycle of watcher activity."""
    if config is None:
        config = WATCHER_CONFIG

    all_detections = []
    autocatalytic_count = 0
    spawned = []
    dissolved = []

    # Each watcher processes events
    for watcher in watchers:
        detections = watcher_detect(watcher, events)
        all_detections.extend(detections)

        # Check for autocatalysis
        if check_autocatalysis(watcher, []):
            autocatalytic_count += 1
            emit_watcher_receipt(watcher, "autocatalysis")

        # Check for dissolution
        if watcher.cycles_since_autocatalysis > config["dissolution_after_cycles"]:
            dissolved.append(dissolve_watcher(watcher))

    # Remove dissolved watchers
    dissolved_ids = {d["watcher_id"] for d in dissolved}
    watchers = [w for w in watchers if w.watcher_id not in dissolved_ids]

    # Check for new spawns from high-signal events
    for event in events:
        if event.fraud_signal >= config["spawn_threshold"]:
            new_watcher = spawn_watcher(
                {"domain": event.domain, "signal": event.fraud_signal},
                config,
                cycle,
            )
            if new_watcher:
                spawned.append(new_watcher)
                watchers.append(new_watcher)

    # Govern population
    entropy_budget = 1.0  # Full entropy budget
    watchers = govern_population(watchers, entropy_budget)

    return {
        "cycle": cycle,
        "active_watchers": len(watchers),
        "detections": len(all_detections),
        "autocatalytic": autocatalytic_count,
        "spawned": len(spawned),
        "dissolved": len(dissolved),
        "watchers": watchers,
    }


def generate_fraud_events(
    n_events: int, domain: str, fraud_rate: float, seed: int
) -> list[FraudEvent]:
    """Generate synthetic fraud events for testing."""
    rng = random.Random(seed)

    events = []
    for i in range(n_events):
        signal = rng.random()
        if rng.random() < fraud_rate:
            signal = rng.uniform(0.6, 1.0)  # Elevated fraud signal

        event = FraudEvent(
            event_id=f"EVENT-{domain[:3].upper()}-{i:05d}",
            domain=domain,
            fraud_signal=signal,
            timestamp=i,
        )
        events.append(event)

    return events


def run_watcher_simulation(
    n_cycles: int = 100,
    n_events_per_cycle: int = 50,
    domains: list | None = None,
    fraud_rate: float = 0.15,
    seed: int = 42,
) -> dict:
    """Run complete watcher simulation."""
    if domains is None:
        domains = ["hope_florida", "insurance", "sheriff", "corporate", "pandemic"]

    rng = random.Random(seed)
    watchers: list[Watcher] = []
    cycle_results = []
    total_spawned = 0
    total_autocatalytic = 0

    for cycle in range(n_cycles):
        # Generate events for this cycle
        all_events = []
        for domain in domains:
            events = generate_fraud_events(
                n_events_per_cycle // len(domains),
                domain,
                fraud_rate,
                seed + cycle * 1000 + domains.index(domain),
            )
            all_events.extend(events)

        # Run watcher cycle
        result = run_watcher_cycle(watchers, all_events, cycle)
        cycle_results.append(result)
        watchers = result["watchers"]
        total_spawned += result["spawned"]
        total_autocatalytic = max(total_autocatalytic, result["autocatalytic"])

    # Final statistics
    final_autocatalytic = sum(1 for w in watchers if w.is_autocatalytic)

    return {
        "n_cycles": n_cycles,
        "total_spawned": total_spawned,
        "final_watchers": len(watchers),
        "final_autocatalytic": final_autocatalytic,
        "peak_autocatalytic": total_autocatalytic,
        "cycle_results": cycle_results[-10:],  # Last 10 cycles
    }
