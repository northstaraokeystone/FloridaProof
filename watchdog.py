#!/usr/bin/env python3
"""FloridaProof Watchdog - System health monitoring.

Usage:
    python watchdog.py --check    Run health check
    python watchdog.py --daemon   Run as daemon (placeholder)
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.core import emit_receipt, TENANT_ID, VERSION


def check_core_health():
    """Check core module functionality."""
    try:
        from src.core import dual_hash, emit_receipt, merkle

        # Test dual_hash
        h = dual_hash("test")
        assert ":" in h, "dual_hash format invalid"

        # Test merkle
        m = merkle([{"a": 1}, {"b": 2}])
        assert ":" in m, "merkle format invalid"

        return True, "Core module healthy"
    except Exception as e:
        return False, f"Core module error: {e}"


def check_domain_health():
    """Check domain modules load correctly."""
    try:
        from src.domains import hope_florida, insurance, sheriff, corporate, pandemic

        # Quick function check
        assert hasattr(hope_florida, "run_detection")
        assert hasattr(insurance, "run_detection")
        assert hasattr(sheriff, "run_detection")
        assert hasattr(corporate, "run_detection")
        assert hasattr(pandemic, "run_detection")

        return True, "Domain modules healthy"
    except Exception as e:
        return False, f"Domain module error: {e}"


def check_simulation_health():
    """Check simulation can run."""
    try:
        from src.sim import run_simulation, SimConfig

        config = SimConfig(n_cycles=5)
        result = run_simulation(config)

        assert result.final_state.cycle > 0, "No cycles completed"
        assert len(result.final_state.receipt_ledger) > 0, "No receipts generated"

        return True, f"Simulation healthy ({result.final_state.cycle} cycles)"
    except Exception as e:
        return False, f"Simulation error: {e}"


def run_health_check():
    """Run all health checks."""
    checks = [
        ("core", check_core_health),
        ("domains", check_domain_health),
        ("simulation", check_simulation_health),
    ]

    all_healthy = True
    results = []

    for name, check_fn in checks:
        healthy, message = check_fn()
        results.append({"check": name, "healthy": healthy, "message": message})
        if not healthy:
            all_healthy = False
        status = "OK" if healthy else "FAIL"
        print(f"[{status}] {name}: {message}")

    # Emit watchdog receipt
    emit_receipt(
        "watchdog",
        {
            "tenant_id": TENANT_ID,
            "version": VERSION,
            "all_healthy": all_healthy,
            "checks": results,
        },
    )

    return all_healthy


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="FloridaProof Watchdog")
    parser.add_argument("--check", action="store_true", help="Run health check")
    parser.add_argument("--daemon", action="store_true", help="Run as daemon (placeholder)")

    args = parser.parse_args()

    if args.check:
        healthy = run_health_check()
        sys.exit(0 if healthy else 1)
    elif args.daemon:
        print("Daemon mode not implemented - use --check for health verification")
        sys.exit(0)
    else:
        # Default: run check
        healthy = run_health_check()
        sys.exit(0 if healthy else 1)


if __name__ == "__main__":
    main()
