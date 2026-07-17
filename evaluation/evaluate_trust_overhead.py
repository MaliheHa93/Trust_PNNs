from __future__ import annotations

import csv
from pathlib import Path
from statistics import mean, pstdev
from time import perf_counter

from demos.common import (
    build_default_orchestrator,
    make_chemical_task,
    make_edge_task,
    make_wetware_task,
)

from core.trust_manager import TrustConfig, TrustManager


RESULTS_DIR = Path("evaluation/results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


ITERATIONS = 30


def save_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError("No rows to save.")

    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def measure_case(case_name: str, task, iterations: int) -> dict:
    orchestrator = build_default_orchestrator()

    trust_manager = TrustManager(
        TrustConfig(
            alpha=0.25,
            beta=0.25,
            gamma=0.25,
            delta=0.25,
            tau_s=5.0,
            theta_accept=0.75,
            theta_reject=0.35,
        )
    )

    orchestration_times: list[float] = []
    trust_times: list[float] = []
    total_times: list[float] = []

    for _ in range(iterations):
        start_total = perf_counter()

        start_orchestration = perf_counter()
        run_result = orchestrator.execute_task(task)
        orchestration_ms = (perf_counter() - start_orchestration) * 1000.0

        if not run_result.success or run_result.invocation is None:
            raise RuntimeError(f"phys-MCP run failed for {case_name}: {run_result.failure_reason}")

        start_trust = perf_counter()
        trust_manager.evaluate_invocation(
            invocation=run_result.invocation,
            telemetry_after=run_result.telemetry_after,
        )
        trust_ms = (perf_counter() - start_trust) * 1000.0

        total_ms = (perf_counter() - start_total) * 1000.0

        orchestration_times.append(orchestration_ms)
        trust_times.append(trust_ms)
        total_times.append(total_ms)

    return {
        "case": case_name,
        "iterations": iterations,
        "orchestration_mean_ms": mean(orchestration_times),
        "orchestration_std_ms": pstdev(orchestration_times),
        "trust_mean_ms": mean(trust_times),
        "trust_std_ms": pstdev(trust_times),
        "total_mean_ms": mean(total_times),
        "total_std_ms": pstdev(total_times),
        "trust_overhead_percent": (mean(trust_times) / mean(orchestration_times)) * 100.0,
    }


def main() -> None:
    cases = [
        ("chemical", make_chemical_task(task_id="overhead-chemical", input_level=1.4)),
        ("wetware", make_wetware_task(task_id="overhead-wetware")),
        ("edge", make_edge_task(task_id="overhead-edge")),
    ]

    rows = []

    for case_name, task in cases:
        row = measure_case(case_name, task, ITERATIONS)
        rows.append(row)

        print(
            f"{case_name}: orchestration={row['orchestration_mean_ms']:.4f} ms, "
            f"trust={row['trust_mean_ms']:.4f} ms, "
            f"trust overhead={row['trust_overhead_percent']:.4f}%"
        )

    output_path = RESULTS_DIR / "trust_overhead_results.csv"
    save_csv(output_path, rows)

    print(f"Saved trust overhead results to: {output_path}")


if __name__ == "__main__":
    main()