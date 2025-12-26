#!/bin/bash
# gate_t24h.sh - T+24h MVP gate check
# RUN THIS OR KILL PROJECT

set -e

echo "=== FloridaProof T+24h Gate Check ==="

# Run T+2h first
./gate_t2h.sh

# Run tests
python -m pytest tests/ -q || { echo "FAIL: tests failed"; exit 1; }

# Check emit_receipt in all src files
for f in src/*.py src/domains/*.py; do
    if [ -f "$f" ] && [ "$(basename $f)" != "__init__.py" ]; then
        grep -q "emit_receipt" "$f" || echo "WARN: $f missing emit_receipt"
    fi
done

# Check assertions in tests
for f in tests/test_*.py; do
    grep -q "assert" "$f" || { echo "FAIL: $f missing assertions"; exit 1; }
done

# Quick simulation test
python -c "from src.sim import run_simulation, SimConfig; r=run_simulation(SimConfig(n_cycles=10)); print(f'Cycles: {r.final_state.cycle}, Receipts: {len(r.final_state.receipt_ledger)}')" || { echo "FAIL: simulation failed"; exit 1; }

echo "PASS: T+24h gate - MVP complete"
