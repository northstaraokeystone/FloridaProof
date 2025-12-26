#!/bin/bash
# gate_t2h.sh - T+2h SKELETON gate check
# RUN THIS OR KILL PROJECT

set -e

echo "=== FloridaProof T+2h Gate Check ==="

# Check required files exist
[ -f spec.md ]            || { echo "FAIL: no spec.md"; exit 1; }
[ -f ledger_schema.json ] || { echo "FAIL: no ledger_schema.json"; exit 1; }
[ -f cli.py ]             || { echo "FAIL: no cli.py"; exit 1; }

# Check CLI emits valid receipt
python cli.py --test 2>&1 | grep -q '"receipt_type"' || { echo "FAIL: cli.py doesn't emit receipt"; exit 1; }

# Check core module exists and has required functions
python -c "from src.core import dual_hash, emit_receipt, merkle, TENANT_ID" || { echo "FAIL: core module incomplete"; exit 1; }

# Verify dual_hash format
python -c "from src.core import dual_hash; h=dual_hash('test'); assert ':' in h, 'Invalid hash format'" || { echo "FAIL: dual_hash format"; exit 1; }

echo "PASS: T+2h gate - SKELETON complete"
