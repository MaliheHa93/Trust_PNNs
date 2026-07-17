# evaluation/evaluate_trust_layer.py

from __future__ import annotations

import csv
import random
from pathlib import Path

from core.trust_manager import TrustManager, TrustConfig


RESULTS_DIR = Path("evaluation/results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def simulate_output(true_value: float, noise_level: float, drift_level: float):
    noise = random.gauss(0, noise_level)
    drift = drift_level
    value = true_value + noise + drift
    return value


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

    rows = []

    true_value = 1.0

    for noise_level in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]:
        for drift_level in [0.0, 0.1, 0.2, 0.3, 0.4]:
            for delay_s in [0.1, 1.0, 3.0, 5.0, 10.0]:

                freshness = trust_manager.compute_freshness(delay_s)

                confidence = max(0.0, 1.0 - noise_level - drift_level)
                noise = min(noise_level, 1.0)
                drift = min(drift_level, 1.0)

                trust = trust_manager.compute_trust(
                    confidence=confidence,
                    noise=noise,
                    drift=drift,
                    freshness=freshness,
                )

                predicted_value = simulate_output(
                    true_value=true_value,
                    noise_level=noise_level,
                    drift_level=drift_level,
                )

                absolute_error = abs(predicted_value - true_value)

                accepted = trust >= trust_manager.config.theta_accept
                bad_output = absolute_error > 0.3
                false_acceptance = accepted and bad_output

                rows.append(
                    {
                        "noise_level": noise_level,
                        "drift_level": drift_level,
                        "delay_s": delay_s,
                        "confidence": confidence,
                        "freshness": freshness,
                        "trust": trust,
                        "predicted_value": predicted_value,
                        "absolute_error": absolute_error,
                        "accepted": accepted,
                        "bad_output": bad_output,
                        "false_acceptance": false_acceptance,
                    }
                )

    output_path = RESULTS_DIR / "trust_layer_results.csv"

    with output_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved results to {output_path}")


if __name__ == "__main__":
    main()