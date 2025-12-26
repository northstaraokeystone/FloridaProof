# FloridaProof Specification

## Overview

FloridaProof v1.0 is a Monte Carlo simulation system for detecting fraud patterns in Florida's 2024-2025 political landscape. Unlike Minnesota's industrial-scale "Feeding Our Future" fraud, Florida's pattern is fund routing through quasi-public foundations into political action.

## Inputs

### Domain Configurations
- `n_settlements`: Number of settlement flows to simulate (default: 100)
- `n_contracts`: Number of sheriff contracts (default: 500)
- `n_donations`: Number of corporate donations (default: 1000)
- `fraud_rates`: Per-domain fraud injection rates
  - hope_florida: 0.15
  - insurance: 0.10
  - sheriff: 0.08
  - corporate: 0.12
  - pandemic: 0.05

### Simulation Parameters
- `n_cycles`: Number of Monte Carlo cycles (default: 10000)
- `random_seed`: Base seed for reproducibility (default: 42)
- `cascade_enabled`: Enable cross-domain cascade (default: true)
- `network_enabled`: Enable protection network mapping (default: true)
- `watchers_enabled`: Enable self-spawning fraud hunters (default: true)
- `pressure_enabled`: Enable political pressure simulation (default: true)

## Outputs

### Detection Rates
- Per-domain fraud detection rates (0.0 - 1.0)
- Aggregate detection rate across all domains
- Precision/recall metrics per domain

### Resilience Metrics
- NEURON alpha at 50% pressure
- NEURON alpha at 75% pressure
- Watcher autocatalysis rate
- Cascade propagation depth

### Receipts
All outputs are emitted as CLAUDEME-compliant receipts:
- `hope_florida_receipt`: Settlement routing detection
- `insurance_influence_receipt`: Donor-vote correlation
- `sheriff_contract_receipt`: Contract fraud detection
- `corporate_capture_receipt`: Policy-donation correlation
- `pandemic_fraud_receipt`: PPP/unemployment fraud
- `cascade_receipt`: Cross-domain propagation
- `network_receipt`: Protection network mapping
- `watcher_receipt`: Self-spawning agent activity
- `pressure_receipt`: Political pressure resilience
- `axiom_receipt`: Compression-based detection
- `simulation_receipt`: Overall simulation results

## SLO Thresholds

| SLO | Threshold | Stoprule Action |
|-----|-----------|-----------------|
| Hope Florida detection | >= 0.92 | emit violation |
| Insurance detection | >= 0.88 | emit violation |
| Sheriff detection | >= 0.90 | emit violation |
| Corporate detection | >= 0.85 | emit violation |
| Pandemic detection | >= 0.95 | emit violation |
| NEURON alpha @ 50% | >= 0.70 | emit violation |
| Watchers autocatalytic | >= 3 | emit violation |
| Cascade traced | true | emit violation |
| Simulation time | < 600s | log warning |
| Memory | < 4GB | halt |

## Stoprules

1. `stoprule_hash_mismatch`: Emit anomaly, halt on hash verification failure
2. `stoprule_invalid_receipt`: Emit anomaly, halt on malformed receipt
3. `stoprule_detection_below_threshold`: Emit violation, continue with alert

## Rollback Strategy

If simulation fails mid-execution:
1. Preserve all emitted receipts to `receipts.jsonl`
2. Log failure state with cycle number
3. Compute merkle root of partial receipt chain
4. Resume from last complete cycle on retry

## 6 Mandatory Scenarios

1. **BASELINE**: Standard parameters, zero violations, detection >= 92%
2. **STRESS**: High fraud (40%), detection >= 75%
3. **GENESIS**: Watcher emergence, >= 10 spawned, >= 3 autocatalytic
4. **CASCADE**: Cross-domain propagation traced
5. **PRESSURE**: NEURON alpha >= 0.70 at 50% disruption
6. **GODEL**: Edge cases handled gracefully
