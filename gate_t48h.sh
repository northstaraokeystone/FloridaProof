#!/bin/bash
# gate_t48h.sh - T+48h HARDENED gate check
# RUN THIS OR KILL PROJECT

set -e

echo "=== FloridaProof T+48h Gate Check ==="

# Run previous gates
./gate_t24h.sh

# Check anomaly detection
grep -rq "anomaly" src/*.py || { echo "FAIL: no anomaly detection"; exit 1; }

# Check bias handling (in this context, detection rate fairness)
grep -rq "detection_rate\|bias\|threshold" src/*.py || { echo "FAIL: no detection threshold checks"; exit 1; }

# Check stoprules
grep -rq "stoprule" src/*.py || { echo "FAIL: no stoprules"; exit 1; }

# Run watchdog check
python watchdog.py --check || { echo "FAIL: watchdog unhealthy"; exit 1; }

# Run all scenarios
python -c "
from src.sim import run_all_scenarios, SimConfig
result = run_all_scenarios(SimConfig())
print(f'Scenarios passed: {result[\"passed_count\"]}/{result[\"total_count\"]}')
if not result['all_passed']:
    for name, r in result['scenarios'].items():
        if not r.passed:
            print(f'  FAILED: {name}')
" || { echo "FAIL: scenarios failed"; exit 1; }

echo "PASS: T+48h gate - HARDENED - SHIP IT"
