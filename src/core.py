"""Core module: CLAUDEME-compliant foundation for FloridaProof.

Every other file imports this. Contains dual_hash, emit_receipt, merkle, StopRule.
"""

import hashlib
import json
from datetime import datetime, timezone
from typing import Any

try:
    import blake3

    HAS_BLAKE3 = True
except ImportError:
    HAS_BLAKE3 = False

# Constants
TENANT_ID = "floridaproof-fl-state"
VERSION = "1.0.0"


def dual_hash(data: bytes | str) -> str:
    """SHA256:BLAKE3 format per CLAUDEME section 8. Pure function."""
    if isinstance(data, str):
        data = data.encode()
    sha = hashlib.sha256(data).hexdigest()
    b3 = blake3.blake3(data).hexdigest() if HAS_BLAKE3 else sha
    return f"{sha}:{b3}"


def emit_receipt(receipt_type: str, data: dict) -> dict:
    """Create receipt with ts, tenant_id, payload_hash. Prints JSON to stdout."""
    receipt = {
        "receipt_type": receipt_type,
        "ts": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "tenant_id": data.get("tenant_id", TENANT_ID),
        "payload_hash": dual_hash(json.dumps(data, sort_keys=True)),
        **data,
    }
    print(json.dumps(receipt), flush=True)
    return receipt


def merkle(items: list) -> str:
    """Compute Merkle root using dual_hash. Handle empty/odd counts."""
    if not items:
        return dual_hash(b"empty")
    hashes = [dual_hash(json.dumps(i, sort_keys=True)) for i in items]
    while len(hashes) > 1:
        if len(hashes) % 2:
            hashes.append(hashes[-1])
        hashes = [dual_hash(hashes[i] + hashes[i + 1]) for i in range(0, len(hashes), 2)]
    return hashes[0]


class StopRuleException(Exception):
    """Raised when stoprule triggers. Never catch silently."""

    pass


def stoprule_hash_mismatch(expected: str, actual: str) -> None:
    """Emit anomaly receipt and halt on hash mismatch."""
    emit_receipt(
        "anomaly",
        {
            "metric": "hash_mismatch",
            "expected": expected,
            "actual": actual,
            "classification": "violation",
            "action": "halt",
        },
    )
    raise StopRuleException(f"Hash mismatch: {expected} != {actual}")


def stoprule_invalid_receipt(reason: str) -> None:
    """Emit anomaly receipt and halt on invalid receipt."""
    emit_receipt(
        "anomaly",
        {
            "metric": "invalid_receipt",
            "reason": reason,
            "classification": "violation",
            "action": "halt",
        },
    )
    raise StopRuleException(f"Invalid receipt: {reason}")


def stoprule_detection_below_threshold(rate: float, threshold: float) -> dict:
    """Emit violation receipt when detection rate below threshold. Continue execution."""
    return emit_receipt(
        "violation",
        {
            "metric": "detection_rate",
            "rate": rate,
            "threshold": threshold,
            "classification": "degradation",
            "action": "alert",
        },
    )
