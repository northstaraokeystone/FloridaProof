"""Tests for core module."""

import json
import pytest
from src.core import (
    dual_hash,
    emit_receipt,
    merkle,
    StopRuleException,
    stoprule_hash_mismatch,
    stoprule_invalid_receipt,
    stoprule_detection_below_threshold,
    TENANT_ID,
    VERSION,
)


class TestDualHash:
    """Tests for dual_hash function."""

    def test_dual_hash_string(self):
        """Test dual_hash with string input."""
        result = dual_hash("test")
        assert ":" in result
        parts = result.split(":")
        assert len(parts) == 2
        assert len(parts[0]) == 64  # SHA256 hex length
        assert len(parts[1]) == 64  # BLAKE3 hex length

    def test_dual_hash_bytes(self):
        """Test dual_hash with bytes input."""
        result = dual_hash(b"test")
        assert ":" in result

    def test_dual_hash_deterministic(self):
        """Test dual_hash is deterministic."""
        result1 = dual_hash("test")
        result2 = dual_hash("test")
        assert result1 == result2

    def test_dual_hash_different_inputs(self):
        """Test dual_hash produces different outputs for different inputs."""
        result1 = dual_hash("test1")
        result2 = dual_hash("test2")
        assert result1 != result2


class TestEmitReceipt:
    """Tests for emit_receipt function."""

    def test_emit_receipt_basic(self, capsys):
        """Test basic receipt emission."""
        receipt = emit_receipt("test", {"key": "value"})

        assert receipt["receipt_type"] == "test"
        assert "ts" in receipt
        assert "tenant_id" in receipt
        assert "payload_hash" in receipt
        assert receipt["key"] == "value"

        # Check stdout output
        captured = capsys.readouterr()
        output = json.loads(captured.out.strip())
        assert output["receipt_type"] == "test"

    def test_emit_receipt_tenant_id(self, capsys):
        """Test receipt includes default tenant_id."""
        receipt = emit_receipt("test", {})
        assert receipt["tenant_id"] == TENANT_ID

    def test_emit_receipt_custom_tenant(self, capsys):
        """Test receipt with custom tenant_id."""
        receipt = emit_receipt("test", {"tenant_id": "custom-tenant"})
        assert receipt["tenant_id"] == "custom-tenant"

    def test_emit_receipt_has_timestamp(self, capsys):
        """Test receipt has valid timestamp."""
        receipt = emit_receipt("test", {})
        assert receipt["ts"].endswith("Z")
        assert "T" in receipt["ts"]


class TestMerkle:
    """Tests for merkle function."""

    def test_merkle_empty(self):
        """Test merkle with empty list."""
        result = merkle([])
        assert ":" in result

    def test_merkle_single(self):
        """Test merkle with single item."""
        result = merkle([{"key": "value"}])
        assert ":" in result

    def test_merkle_multiple(self):
        """Test merkle with multiple items."""
        items = [{"key": str(i)} for i in range(10)]
        result = merkle(items)
        assert ":" in result

    def test_merkle_deterministic(self):
        """Test merkle is deterministic."""
        items = [{"key": "value1"}, {"key": "value2"}]
        result1 = merkle(items)
        result2 = merkle(items)
        assert result1 == result2

    def test_merkle_odd_count(self):
        """Test merkle handles odd number of items."""
        items = [{"key": str(i)} for i in range(7)]
        result = merkle(items)
        assert ":" in result


class TestStoprules:
    """Tests for stoprule functions."""

    def test_stoprule_hash_mismatch(self, capsys):
        """Test hash mismatch stoprule raises exception."""
        with pytest.raises(StopRuleException) as exc_info:
            stoprule_hash_mismatch("expected", "actual")

        assert "Hash mismatch" in str(exc_info.value)

        # Check anomaly receipt was emitted
        captured = capsys.readouterr()
        output = json.loads(captured.out.strip())
        assert output["receipt_type"] == "anomaly"

    def test_stoprule_invalid_receipt(self, capsys):
        """Test invalid receipt stoprule raises exception."""
        with pytest.raises(StopRuleException) as exc_info:
            stoprule_invalid_receipt("test reason")

        assert "Invalid receipt" in str(exc_info.value)

    def test_stoprule_detection_below_threshold(self, capsys):
        """Test detection below threshold stoprule emits violation."""
        result = stoprule_detection_below_threshold(0.5, 0.9)

        assert result["receipt_type"] == "violation"
        assert result["rate"] == 0.5
        assert result["threshold"] == 0.9


class TestConstants:
    """Tests for module constants."""

    def test_tenant_id(self):
        """Test TENANT_ID is set correctly."""
        assert TENANT_ID == "floridaproof-fl-state"

    def test_version(self):
        """Test VERSION is set correctly."""
        assert VERSION == "1.0.0"
