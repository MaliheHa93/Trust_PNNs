from __future__ import annotations

import csv
import random
from pathlib import Path
from time import perf_counter
from types import SimpleNamespace

from core.trust_manager import TrustConfig, TrustManager
from core.fog_fusion import FogFusionEngine
from core.trust_models import EvidenceRecord
from evaluation.experiment_config import ExperimentConfig
from evaluation.scenario_generator import make_random_scenario


RESULTS_DIR = Path("evaluation/results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def save_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError("No rows to save.")

    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def make_invocation_from_scenario(scenario):
    return SimpleNamespace(
        backend_id="synthetic-pnn",
        task_id=f"scenario-{scenario.scenario_id}",
        output_payload={
            "value": scenario.predicted_value,
            "modality": "synthetic_numeric",
        },
        confidence=scenario.confidence,
        execution_latency_ms=scenario.delay_s * 1000.0,
        backend_state="ready",
        notes=None,
    )


def main() -> None:
    cfg = ExperimentConfig()
    random.seed(cfg.seed)

    trust_manager = TrustManager(
        TrustConfig(
            alpha=cfg.alpha,
            beta=cfg.beta,
            gamma=cfg.gamma,
            delta=cfg.delta,
            tau_s=cfg.tau_s,
            theta_accept=cfg.theta_accept,
            theta_reject=cfg.theta_reject,
        )
    )

    fusion_engine = FogFusionEngine()

    rows: list[dict] = []

    for scenario_count in cfg.scalability_scenarios:
        scenarios = [
            make_random_scenario(
                scenario_id=i,
                true_value=cfg.true_value,
                error_threshold=cfg.error_threshold,
                correlated_confidence=True,
            )
            for i in range(scenario_count)
        ]

        start = perf_counter()

        for scenario in scenarios:
            invocation = make_invocation_from_scenario(scenario)

            telemetry = {
                "noise_score": scenario.noise,
                "drift_score": scenario.drift,
                "age_of_information_ms": scenario.delay_s * 1000.0,
                "output_modality": "synthetic_numeric",
            }

            trust_manager.evaluate_invocation(
                invocation=invocation,
                telemetry_after=telemetry,
            )

        total_ms = (perf_counter() - start) * 1000.0

        rows.append(
            {
                "experiment": "edge_trust_scalability",
                "scenario_count": scenario_count,
                "source_count": None,
                "total_time_ms": total_ms,
                "time_per_record_ms": total_ms / scenario_count,
            }
        )

    for source_count in cfg.source_counts:
        records: list[EvidenceRecord] = []

        for source_id in range(source_count):
            scenario = make_random_scenario(
                scenario_id=source_id,
                true_value=cfg.true_value,
                error_threshold=cfg.error_threshold,
                correlated_confidence=True,
            )

            freshness = trust_manager.compute_freshness(scenario.delay_s)
            trust = trust_manager.compute_trust(
                confidence=scenario.confidence,
                noise=scenario.noise,
                drift=scenario.drift,
                freshness=freshness,
            )

            records.append(
                EvidenceRecord(
                    backend_id=f"source-{source_id}",
                    task_id="fusion-scalability-task",
                    value=scenario.predicted_value,
                    confidence=scenario.confidence,
                    noise=scenario.noise,
                    drift=scenario.drift,
                    timestamp=0.0,
                    modality="synthetic_numeric",
                    provenance={"source_id": source_id},
                    freshness=freshness,
                    trust=trust,
                )
            )

        start = perf_counter()
        fusion_engine.fuse_numeric(records)
        total_ms = (perf_counter() - start) * 1000.0

        rows.append(
            {
                "experiment": "fog_fusion_scalability",
                "scenario_count": None,
                "source_count": source_count,
                "total_time_ms": total_ms,
                "time_per_record_ms": total_ms / source_count,
            }
        )

    output_path = RESULTS_DIR / "scalability_results.csv"
    save_csv(output_path, rows)

    print(f"Saved scalability results to: {output_path}")


if __name__ == "__main__":
    main()