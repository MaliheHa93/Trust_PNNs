from __future__ import annotations

import csv
import random
from pathlib import Path
from types import SimpleNamespace

from demos.common import build_default_orchestrator, make_chemical_task
from core.trust_manager import TrustConfig, TrustManager


RESULTS_DIR = Path("evaluation/results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def save_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError("No rows to save.")

    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def decide_from_score(score: float, theta_accept: float, theta_reject: float) -> str:
    if score >= theta_accept:
        return "accept"
    if score < theta_reject:
        return "reject"
    return "forward_to_fog"


def make_modified_invocation(
    original_invocation,
    value: float,
    confidence: float,
    delay_s: float,
):
    return SimpleNamespace(
        backend_id=getattr(original_invocation, "backend_id", "chemical-backend"),
        task_id=getattr(original_invocation, "task_id", "random-scenario-task"),
        output_payload={
            "value": value,
            "modality": "chemical_concentration",
        },
        confidence=confidence,
        execution_latency_ms=delay_s * 1000.0,
        backend_state=getattr(original_invocation, "backend_state", "ready"),
        notes=getattr(original_invocation, "notes", None),
    )


def main() -> None:
    random.seed(42)

    orchestrator = build_default_orchestrator()
    task = make_chemical_task(task_id="random-scenario-chemical", input_level=1.8)

    base_run = orchestrator.execute_task(task)

    if not base_run.success or base_run.invocation is None:
        raise RuntimeError(f"phys-MCP invocation failed: {base_run.failure_reason}")

    theta_accept = 0.75
    theta_reject = 0.35

    trust_manager = TrustManager(
        TrustConfig(
            alpha=0.25,
            beta=0.25,
            gamma=0.25,
            delta=0.25,
            tau_s=5.0,
            theta_accept=theta_accept,
            theta_reject=theta_reject,
        )
    )

    true_value = 1.0
    error_threshold = 0.30

    number_of_scenarios = 10000
    rows: list[dict] = []

    for scenario_id in range(number_of_scenarios):
        confidence = random.uniform(0.0, 1.0)
        noise_level = random.uniform(0.0, 1.0)
        drift_level = random.uniform(0.0, 1.0)
        delay_s = random.uniform(0.0, 20.0)

        random_noise = random.gauss(0.0, noise_level)
        predicted_value = true_value + random_noise + drift_level
        absolute_error = abs(predicted_value - true_value)

        bad_output = absolute_error > error_threshold

        invocation = make_modified_invocation(
            original_invocation=base_run.invocation,
            value=predicted_value,
            confidence=confidence,
            delay_s=delay_s,
        )

        telemetry = dict(base_run.telemetry_after)
        telemetry.update(
            {
                "noise_score": noise_level,
                "drift_score": drift_level,
                "age_of_information_ms": delay_s * 1000.0,
                "output_modality": "chemical_concentration",
            }
        )

        trust_result = trust_manager.evaluate_invocation(
            invocation=invocation,
            telemetry_after=telemetry,
        )

        freshness = trust_result.evidence.freshness
        full_trust_score = trust_result.evidence.trust

        methods = {
            "raw_phys_mcp_output": {
                "decision": "accept",
                "method_score": None,
            },
            "confidence_only": {
                "decision": decide_from_score(
                    confidence,
                    theta_accept,
                    theta_reject,
                ),
                "method_score": confidence,
            },
            "freshness_only": {
                "decision": decide_from_score(
                    freshness,
                    theta_accept,
                    theta_reject,
                ),
                "method_score": freshness,
            },
            "full_trust_score": {
                "decision": trust_result.decision.value,
                "method_score": full_trust_score,
            },
        }

        for method_name, method_info in methods.items():
            decision = method_info["decision"]
            method_score = method_info["method_score"]

            accepted = decision == "accept"
            rejected = decision == "reject"
            forwarded = decision == "forward_to_fog"

            false_acceptance = accepted and bad_output
            false_rejection = rejected and not bad_output

            rows.append(
                {
                    "scenario_id": scenario_id,
                    "method": method_name,
                    "confidence": confidence,
                    "noise_level": noise_level,
                    "drift_level": drift_level,
                    "delay_s": delay_s,
                    "freshness": freshness,
                    "full_trust_score": full_trust_score,
                    "method_score": method_score,
                    "true_value": true_value,
                    "predicted_value": predicted_value,
                    "absolute_error": absolute_error,
                    "bad_output": bad_output,
                    "decision": decision,
                    "accepted": accepted,
                    "rejected": rejected,
                    "forwarded": forwarded,
                    "false_acceptance": false_acceptance,
                    "false_rejection": false_rejection,
                }
            )

    output_path = RESULTS_DIR / "random_scenario_results.csv"
    save_csv(output_path, rows)

    print(f"Saved random scenario results to: {output_path}")


if __name__ == "__main__":
    main()