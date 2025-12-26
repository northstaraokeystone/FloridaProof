#!/usr/bin/env python3
"""FloridaProof CLI - Florida Political Fund Routing Detection Monte Carlo.

Usage:
    python cli.py --test           Run quick validation test
    python cli.py --scenario NAME  Run specific scenario (BASELINE, STRESS, etc.)
    python cli.py --all            Run all 6 mandatory scenarios
    python cli.py --sim N          Run N-cycle simulation
"""

import argparse
import json
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.core import emit_receipt, dual_hash, TENANT_ID, VERSION
from src.sim import SimConfig, run_simulation, run_scenario, run_all_scenarios


def run_test():
    """Run quick validation test."""
    # Emit a test receipt
    receipt = emit_receipt(
        "cli_test",
        {
            "tenant_id": TENANT_ID,
            "version": VERSION,
            "status": "operational",
        },
    )
    return receipt


def run_quick_sim(n_cycles: int = 100):
    """Run a quick simulation."""
    config = SimConfig(n_cycles=n_cycles)
    result = run_simulation(config)

    summary = {
        "cycles_completed": result.final_state.cycle,
        "detection_rates": result.detection_rates,
        "all_passed": result.all_passed,
        "receipts_generated": len(result.final_state.receipt_ledger),
        "receipt_hash": result.receipt_hash[:32] + "..." if result.receipt_hash else "",
    }

    emit_receipt("simulation_complete", summary)
    return summary


def run_single_scenario(scenario: str):
    """Run a single scenario."""
    config = SimConfig()
    result = run_scenario(scenario, config)

    summary = {
        "scenario": result.scenario,
        "passed": result.passed,
        "detection_rates": result.detection_rates,
        "alpha": result.alpha,
        "details": result.details,
    }

    emit_receipt("scenario_complete", summary)
    return summary


def run_all():
    """Run all 6 mandatory scenarios."""
    config = SimConfig()
    result = run_all_scenarios(config)

    summary = {
        "all_passed": result["all_passed"],
        "passed_count": result["passed_count"],
        "total_count": result["total_count"],
        "scenarios": {
            name: {"passed": r.passed, "alpha": r.alpha}
            for name, r in result["scenarios"].items()
        },
    }

    emit_receipt("all_scenarios_complete", summary)
    return summary


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="FloridaProof: Florida Political Fund Routing Detection Monte Carlo"
    )
    parser.add_argument(
        "--test", action="store_true", help="Run quick validation test"
    )
    parser.add_argument(
        "--scenario",
        type=str,
        choices=["BASELINE", "STRESS", "GENESIS", "CASCADE", "PRESSURE", "GODEL"],
        help="Run specific scenario",
    )
    parser.add_argument(
        "--all", action="store_true", help="Run all 6 mandatory scenarios"
    )
    parser.add_argument(
        "--sim", type=int, default=0, help="Run N-cycle simulation"
    )

    args = parser.parse_args()

    if args.test:
        run_test()
    elif args.scenario:
        run_single_scenario(args.scenario)
    elif args.all:
        run_all()
    elif args.sim > 0:
        run_quick_sim(args.sim)
    else:
        # Default: run test
        run_test()


if __name__ == "__main__":
    main()
