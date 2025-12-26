"""Cross-Domain Cascade Engine.

Fund routing in one domain propagates to others.
Pattern from WaProof: "fraud is COUPLED".
"""

from dataclasses import dataclass, field
from typing import Any

from .core import TENANT_ID, emit_receipt

# Cascade weight matrix (Florida-specific)
CASCADE_WEIGHTS = {
    ("hope_florida", "corporate"): 0.75,  # Foundation donors overlap with corporate
    ("hope_florida", "insurance"): 0.40,  # Insurance industry settlement routing
    ("corporate", "insurance"): 0.65,  # Shared lobbying networks
    ("sheriff", "corporate"): 0.30,  # Contract awards to donor-linked firms
    ("pandemic", "sheriff"): 0.20,  # PPP funds through sheriff contracts
    ("insurance", "corporate"): 0.55,  # Insurer donations to same PACs
    ("hope_florida", "sheriff"): 0.25,  # Indirect county connections
    ("corporate", "pandemic"): 0.15,  # Corporate PPP abuse
}


@dataclass
class CascadeEvent:
    """Represents a cascade event across domains."""

    source_domain: str
    affected_domains: list = field(default_factory=list)
    cascade_path: list = field(default_factory=list)
    total_propagation: float = 0.0
    systemic_risk: float = 0.0


def build_cascade_graph(weights: dict | None = None) -> dict:
    """Construct directed graph of domain interactions."""
    if weights is None:
        weights = CASCADE_WEIGHTS

    graph = {}
    all_domains = set()

    for (source, target), weight in weights.items():
        all_domains.add(source)
        all_domains.add(target)

        if source not in graph:
            graph[source] = {}
        graph[source][target] = weight

    # Ensure all domains have entries
    for domain in all_domains:
        if domain not in graph:
            graph[domain] = {}

    return graph


def propagate_fraud_signal(
    graph: dict, source: str, signal: float, visited: set | None = None
) -> dict:
    """Propagate fraud signal through cascade network."""
    if visited is None:
        visited = set()

    results = {source: signal}
    visited.add(source)

    if source not in graph:
        return results

    for target, weight in graph[source].items():
        if target not in visited:
            propagated_signal = signal * weight
            if propagated_signal > 0.05:  # Threshold for meaningful propagation
                results[target] = propagated_signal
                # Recursive propagation with decaying signal
                sub_results = propagate_fraud_signal(
                    graph, target, propagated_signal * 0.8, visited.copy()
                )
                for sub_target, sub_signal in sub_results.items():
                    if sub_target not in results or sub_signal > results[sub_target]:
                        results[sub_target] = sub_signal

    return results


def detect_cascade_event(signals: dict, threshold: float = 0.3) -> list[CascadeEvent]:
    """Identify when fraud cascades across multiple domains."""
    events = []

    # Find domains with elevated signals
    elevated = {domain: sig for domain, sig in signals.items() if sig >= threshold}

    if len(elevated) < 2:
        return events

    # Create cascade event
    source = max(elevated, key=elevated.get)
    affected = [d for d in elevated if d != source]

    event = CascadeEvent(
        source_domain=source,
        affected_domains=affected,
        total_propagation=sum(elevated.values()),
        systemic_risk=len(elevated) / len(signals) if signals else 0,
    )

    events.append(event)
    return events


def trace_cascade_path(
    graph: dict, source: str, target: str, path: list | None = None, visited: set | None = None
) -> list:
    """Reconstruct the cascade pathway for audit."""
    if path is None:
        path = []
    if visited is None:
        visited = set()

    path = path + [source]
    visited.add(source)

    if source == target:
        return path

    if source not in graph:
        return []

    for next_node in graph[source]:
        if next_node not in visited:
            new_path = trace_cascade_path(graph, next_node, target, path, visited.copy())
            if new_path:
                return new_path

    return []


def build_cascade_path_with_weights(graph: dict, source: str, target: str) -> list:
    """Build cascade path with weight annotations."""
    path = trace_cascade_path(graph, source, target)

    if len(path) < 2:
        return []

    weighted_path = []
    for i in range(len(path) - 1):
        from_node = path[i]
        to_node = path[i + 1]
        weight = graph.get(from_node, {}).get(to_node, 0)
        weighted_path.append({"from": from_node, "to": to_node, "weight": weight})

    return weighted_path


def compute_systemic_risk(cascade_events: list[CascadeEvent]) -> float:
    """Aggregate risk when multiple cascades active."""
    if not cascade_events:
        return 0.0

    # Total domains affected
    all_affected = set()
    for event in cascade_events:
        all_affected.add(event.source_domain)
        all_affected.update(event.affected_domains)

    # Risk based on spread and propagation
    total_propagation = sum(e.total_propagation for e in cascade_events)
    domain_coverage = len(all_affected) / 5  # 5 fraud domains total

    return min(1.0, (domain_coverage * 0.6) + (total_propagation * 0.1))


def simulate_cascade(
    graph: dict,
    source_domain: str,
    initial_signal: float,
    threshold: float = 0.3,
) -> dict:
    """Run a complete cascade simulation."""
    signals = propagate_fraud_signal(graph, source_domain, initial_signal)
    events = detect_cascade_event(signals, threshold)
    systemic_risk = compute_systemic_risk(events)

    # Build path information
    paths = []
    for event in events:
        for affected in event.affected_domains:
            path = build_cascade_path_with_weights(graph, event.source_domain, affected)
            if path:
                paths.append(path)

    return {
        "signals": signals,
        "events": events,
        "systemic_risk": systemic_risk,
        "paths": paths,
        "domains_affected": len(signals),
    }


def emit_cascade_receipt(cascade_result: dict) -> dict:
    """Emit receipt for cascade detection."""
    events = cascade_result.get("events", [])
    source = events[0].source_domain if events else "unknown"
    affected = events[0].affected_domains if events else []

    return emit_receipt(
        "cascade",
        {
            "tenant_id": TENANT_ID,
            "source_domain": source,
            "affected_domains": affected,
            "cascade_path": cascade_result.get("paths", [])[:5],  # Limit for receipt size
            "total_propagation": sum(cascade_result.get("signals", {}).values()),
            "systemic_risk": cascade_result.get("systemic_risk", 0),
        },
    )


def run_cascade_analysis(
    domain_signals: dict[str, float] | None = None, seed: int = 42
) -> dict:
    """Run cascade analysis on current domain signals."""
    graph = build_cascade_graph()

    if domain_signals is None:
        # Default elevated Hope Florida signal
        domain_signals = {
            "hope_florida": 0.8,
            "insurance": 0.3,
            "sheriff": 0.2,
            "corporate": 0.4,
            "pandemic": 0.1,
        }

    # Find highest signal as source
    source_domain = max(domain_signals, key=domain_signals.get)
    initial_signal = domain_signals[source_domain]

    result = simulate_cascade(graph, source_domain, initial_signal)
    receipt = emit_cascade_receipt(result)

    return {
        "graph": graph,
        "initial_signals": domain_signals,
        "cascade_result": result,
        "receipt": receipt,
    }
