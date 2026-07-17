from __future__ import annotations

import csv
import random
from pathlib import Path
from types import SimpleNamespace

from core.trust_manager import TrustConfig, TrustManager


RESULTS_DIR = Path("evaluation/results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def simulate_predicted_value(
    true_value: float,
    noise_level: float,
    drift_level: float,
) -> float:
    noise = random.gauss(0.0, noise_level)
    drift = drift_level
    return true_value + noise + drift


def main() -> None:
    random.seed(42)

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

    true_value = 1.0
    rows = []

    noise_values = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
    drift_values = [0.0, 0.1, 0.2, 0.3, 0.4]
    delay_values_s = [0.1, 1.0, 3.0, 5.0, 10.0]

    for noise_level in noise_values:
        for drift_level in drift_values:
            for delay_s in delay_values_s:
                predicted_value = simulate_predicted_value(
                    true_value=true_value,
                    noise_level=noise_level,
                    drift_level=drift_level,
                )

                confidence = max(0.0, 1.0 - noise_level - drift_level)

                invocation = SimpleNamespace(
                    backend_id="synthetic-pnn",
                    task_id="synthetic-task",
                    output_payload={
                        "value": predicted_value,
                        "modality": "synthetic_numeric",
                    },
                    confidence=confidence,
                    execution_latency_ms=delay_s * 1000.0,
                    backend_state="ready",
                    notes=None,
                )

                telemetry = {
                    "noise_score": noise_level,
                    "drift_score": drift_level,
                    "age_of_information_ms": delay_s * 1000.0,
                    "output_modality": "synthetic_numeric",
                }

                trust_result = trust_manager.evaluate_invocation(
                    invocation=invocation,
                    telemetry_after=telemetry,
                )

                absolute_error = abs(predicted_value - true_value)
                bad_output = absolute_error > 0.3
                accepted = trust_result.decision.value == "accept"
                rejected = trust_result.decision.value == "reject"
                forwarded = trust_result.decision.value == "forward_to_fog"

                false_acceptance = accepted and bad_output
                false_rejection = rejected and not bad_output

                rows.append(
                    {
                        "noise_level": noise_level,
                        "drift_level": drift_level,
                        "delay_s": delay_s,
                        "predicted_value": predicted_value,
                        "true_value": true_value,
                        "absolute_error": absolute_error,
                        "confidence": confidence,
                        "freshness": trust_result.evidence.freshness,
                        "trust": trust_result.evidence.trust,
                        "decision": trust_result.decision.value,
                        "bad_output": bad_output,
                        "accepted": accepted,
                        "rejected": rejected,
                        "forwarded": forwarded,
                        "false_acceptance": false_acceptance,
                        "false_rejection": false_rejection,
                    }
                )

    output_file = RESULTS_DIR / "trust_layer_results.csv"

    with output_file.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved results to: {output_file}")


if __name__ == "__main__":
    main()