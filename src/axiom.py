"""AXIOM Compression-Based Fraud Detection.

Pattern from ClaimProof: Legitimate fund flows have causal structure - they compress well.
Fraudulent routing breaks this structure - high MDL, low compression ratio.

The Physics: Information theory meets fraud detection.
"""

import json
import zlib
from dataclasses import dataclass, field
from typing import Any

from .core import TENANT_ID, dual_hash, emit_receipt


@dataclass
class FundFlow:
    """Represents a fund flow for compression analysis."""

    flow_id: str
    source: str
    destination: str
    amount: float
    intermediaries: list = field(default_factory=list)
    timestamps: list = field(default_factory=list)
    metadata: dict = field(default_factory=dict)


@dataclass
class CompressionResult:
    """Result of compression analysis."""

    original_bits: int
    compressed_bits: int
    mdl: float
    compression_ratio: float
    fraud_score: float
    structural_anomaly: bool


# Baseline MDL for legitimate flows (empirically determined)
BASELINE_MDL = 100.0
MDL_THRESHOLD_MULTIPLIER = 2.0
COMPRESSION_RATIO_THRESHOLD = 0.5
FRAUD_SCORE_THRESHOLD = 0.8


def compress_fund_flow(flow: FundFlow) -> dict:
    """Attempt to compress fund routing chain."""
    # Serialize flow to bytes
    flow_data = {
        "flow_id": flow.flow_id,
        "source": flow.source,
        "destination": flow.destination,
        "amount": flow.amount,
        "intermediaries": flow.intermediaries,
        "timestamps": flow.timestamps,
        "metadata": flow.metadata,
    }

    original_bytes = json.dumps(flow_data, sort_keys=True).encode()
    original_bits = len(original_bytes) * 8

    # Compress using zlib (approximates Kolmogorov complexity)
    compressed_bytes = zlib.compress(original_bytes, level=9)
    compressed_bits = len(compressed_bytes) * 8

    return {
        "original_bytes": original_bytes,
        "compressed_bytes": compressed_bytes,
        "original_bits": original_bits,
        "compressed_bits": compressed_bits,
    }


def compute_mdl(flow: FundFlow, compressed: dict) -> float:
    """Minimum Description Length of flow.

    MDL = length of compressed representation + length of model description.
    Higher MDL indicates less structure/more randomness.
    """
    compressed_bits = compressed["compressed_bits"]

    # Model complexity: number of intermediaries and timestamps
    model_bits = (
        len(flow.intermediaries) * 64  # Each intermediary adds complexity
        + len(flow.timestamps) * 32  # Timestamps add temporal complexity
        + len(flow.metadata) * 48  # Metadata adds structural complexity
    )

    # Total MDL
    mdl = compressed_bits + model_bits

    return mdl


def compute_compression_ratio(original: int, compressed: int) -> float:
    """Ratio of compressed to original size."""
    if original <= 0:
        return 1.0
    return compressed / original


def fraud_score_from_compression(mdl: float, ratio: float) -> float:
    """High MDL + low ratio = high fraud probability.

    Legitimate flows:
    - Low MDL (simple structure)
    - High compression ratio (predictable patterns)

    Fraudulent flows:
    - High MDL (complex/random structure)
    - Low compression ratio (unpredictable patterns)
    """
    # Normalize MDL relative to baseline
    mdl_factor = mdl / BASELINE_MDL
    mdl_score = min(1.0, max(0.0, (mdl_factor - 1.0) / 2.0))

    # Invert ratio (low ratio = high score)
    ratio_score = 1.0 - ratio

    # Combined score with weights
    fraud_score = mdl_score * 0.6 + ratio_score * 0.4

    return min(1.0, max(0.0, fraud_score))


def detect_structural_anomaly(flow: FundFlow, baseline_mdl: float) -> bool:
    """Does flow deviate from baseline compression?"""
    compressed = compress_fund_flow(flow)
    mdl = compute_mdl(flow, compressed)

    return mdl > baseline_mdl * MDL_THRESHOLD_MULTIPLIER


def analyze_flow(flow: FundFlow, baseline_mdl: float | None = None) -> CompressionResult:
    """Complete compression analysis of a fund flow."""
    if baseline_mdl is None:
        baseline_mdl = BASELINE_MDL

    compressed = compress_fund_flow(flow)
    mdl = compute_mdl(flow, compressed)
    ratio = compute_compression_ratio(
        compressed["original_bits"], compressed["compressed_bits"]
    )
    fraud_score = fraud_score_from_compression(mdl, ratio)
    structural_anomaly = mdl > baseline_mdl * MDL_THRESHOLD_MULTIPLIER

    return CompressionResult(
        original_bits=compressed["original_bits"],
        compressed_bits=compressed["compressed_bits"],
        mdl=mdl,
        compression_ratio=ratio,
        fraud_score=fraud_score,
        structural_anomaly=structural_anomaly,
    )


def generate_legitimate_flow(seed: int) -> FundFlow:
    """Generate a legitimate fund flow with good structure."""
    import random

    rng = random.Random(seed)

    return FundFlow(
        flow_id=f"LEGIT-{seed:05d}",
        source="Settlement Fund",
        destination="Program Beneficiary",
        amount=rng.uniform(10000, 100000),
        intermediaries=["State Treasury"],  # Simple, direct path
        timestamps=[1000, 1001],  # Sequential
        metadata={"purpose": "direct_benefit"},
    )


def generate_fraudulent_flow(seed: int) -> FundFlow:
    """Generate a fraudulent fund flow with broken structure."""
    import random

    rng = random.Random(seed)

    # Complex, convoluted path
    intermediaries = [
        f"Shell-{rng.randint(1, 100)}",
        f"Foundation-{rng.randint(1, 50)}",
        f"Nonprofit-{rng.randint(1, 30)}",
        f"PAC-{rng.randint(1, 20)}",
        f"Entity-{rng.randint(1, 100)}",
    ]

    # Random timestamps (non-sequential)
    timestamps = sorted([rng.randint(1, 1000) for _ in range(5)])

    # Complex metadata
    metadata = {
        f"field_{i}": rng.random() for i in range(10)
    }

    return FundFlow(
        flow_id=f"FRAUD-{seed:05d}",
        source="Settlement Fund",
        destination=f"Unknown-{rng.randint(1, 1000)}",
        amount=rng.uniform(100000, 1000000),
        intermediaries=intermediaries,
        timestamps=timestamps,
        metadata=metadata,
    )


def emit_axiom_receipt(flow: FundFlow, result: CompressionResult) -> dict:
    """Emit receipt for axiom analysis."""
    return emit_receipt(
        "axiom",
        {
            "tenant_id": TENANT_ID,
            "flow_id": flow.flow_id,
            "original_bits": result.original_bits,
            "compressed_bits": result.compressed_bits,
            "mdl": result.mdl,
            "compression_ratio": result.compression_ratio,
            "fraud_score": result.fraud_score,
            "structural_anomaly": result.structural_anomaly,
        },
    )


def run_axiom_analysis(n_flows: int = 100, fraud_rate: float = 0.15, seed: int = 42) -> dict:
    """Run AXIOM analysis on generated flows."""
    import random

    rng = random.Random(seed)

    flows = []
    results = []
    receipts = []

    true_positives = 0
    false_positives = 0
    true_negatives = 0
    false_negatives = 0

    for i in range(n_flows):
        is_fraud = rng.random() < fraud_rate

        if is_fraud:
            flow = generate_fraudulent_flow(seed + i)
        else:
            flow = generate_legitimate_flow(seed + i)

        result = analyze_flow(flow)
        receipt = emit_axiom_receipt(flow, result)

        flows.append(flow)
        results.append(result)
        receipts.append(receipt)

        # Detection based on fraud score threshold
        detected = result.fraud_score > FRAUD_SCORE_THRESHOLD

        if is_fraud and detected:
            true_positives += 1
        elif is_fraud and not detected:
            false_negatives += 1
        elif not is_fraud and detected:
            false_positives += 1
        else:
            true_negatives += 1

    total_fraud = true_positives + false_negatives
    detection_rate = true_positives / total_fraud if total_fraud > 0 else 1.0

    return {
        "n_flows": n_flows,
        "fraud_rate": fraud_rate,
        "true_positives": true_positives,
        "false_positives": false_positives,
        "true_negatives": true_negatives,
        "false_negatives": false_negatives,
        "detection_rate": detection_rate,
        "precision": true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0.0,
        "avg_mdl": sum(r.mdl for r in results) / len(results) if results else 0,
        "avg_ratio": sum(r.compression_ratio for r in results) / len(results) if results else 0,
        "receipts": receipts,
    }
