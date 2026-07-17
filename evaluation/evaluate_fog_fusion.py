from __future__ import annotations

import csv
import random
from pathlib import Path

from core.fog_fusion import FogFusionEngine
from core.trust_models import EvidenceRecord
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


def simple_average(values: list[float]) -> float:
    return sum(values) / len(values)


def confidence_weighted_average(values: list[float], confidences: list[float]) -> float:
    denominator = sum(confidences)

    if denominator == 0:
        return simple_average(values)

    return sum(v * c for v, c in zip(values, confidences)) / denominator


def main() -> None:
    random.seed(42)

    true_value = 1.0
    source_counts = [2, 3, 5, 7]
    noise_levels = [0.05, 0.1, 0.2, 0.3, 0.4]
    drift_levels = [0.0, 0.1, 0.2, 0.3]

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

    fusion_engine = FogFusionEngine()

    rows: list[dict] = []

    for source_count in source_counts:
        for noise_level in noise_levels:
            for drift_level in drift_levels:
                for repetition in range(50):
                    records: list[EvidenceRecord] = []
                    values: list[float] = []
                    confidences: list[float] = []

                    for source_index in range(source_count):
                        source_noise = noise_level * (1.0 + 0.25 * source_index)
                        source_drift = drift_level * (source_index / max(source_count - 1, 1))
                        delay_s = random.choice([0.1, 1.0, 3.0, 5.0])

                        value = true_value + random.gauss(0.0, source_noise) + source_drift
                        confidence = max(0.0, min(1.0, 1.0 - source_noise - source_drift))
                        freshness = trust_manager.compute_freshness(delay_s)

                        trust = trust_manager.compute_trust(
                            confidence=confidence,
                            noise=min(source_noise, 1.0),
                            drift=min(source_drift, 1.0),
                            freshness=freshness,
                        )

                        record = EvidenceRecord(
                            backend_id=f"synthetic-pnn-{source_index}",
                            task_id="fusion-task",
                            value=value,
                            confidence=confidence,
                            noise=min(source_noise, 1.0),
                            drift=min(source_drift, 1.0),
                            timestamp=0.0,
                            modality="synthetic_numeric",
                            provenance={"source_index": source_index},
                            freshness=freshness,
                            trust=trust,
                        )

                        records.append(record)
                        values.append(value)
                        confidences.append(confidence)

                    simple_result = simple_average(values)
                    confidence_result = confidence_weighted_average(values, confidences)
                    trust_result = fusion_engine.fuse_numeric(records)

                    rows.append(
                        {
                            "source_count": source_count,
                            "noise_level": noise_level,
                            "drift_level": drift_level,
                            "repetition": repetition,
                            "simple_average_error": abs(simple_result - true_value),
                            "confidence_weighted_error": abs(confidence_result - true_value),
                            "trust_aware_error": abs(trust_result - true_value),
                            "simple_average_result": simple_result,
                            "confidence_weighted_result": confidence_result,
                            "trust_aware_result": trust_result,
                        }
                    )

    output_path = RESULTS_DIR / "fog_fusion_results.csv"
    save_csv(output_path, rows)

    print(f"Saved fog fusion results to: {output_path}")


if __name__ == "__main__":
    main()