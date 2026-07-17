from __future__ import annotations

import csv
import random
from pathlib import Path
from types import SimpleNamespace

from core.trust_manager import TrustConfig, TrustManager
from evaluation.experiment_config import ExperimentConfig
from evaluation.scenario_generator import make_random_scenario
from evaluation.metrics import summarize_decisions


RESULTS_DIR = Path("evaluation/results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def save_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError("No rows to save.")

    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def decide_from_trust(score: float, theta_accept: float, theta_reject: float) -> str:
    if score >= theta_accept:
        return "accept"
    if score < theta_reject:
        return "reject"
    return "forward_to_fog"


def make_invocation(scenario):
    return SimpleNamespace(
        backend_id="synthetic-pnn",
        task_id=f"ablation-{scenario.scenario_id}",
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

    scenarios = [
        make_random_scenario(
            scenario_id=i,
            true_value=cfg.true_value,
            error_threshold=cfg.error_threshold,
            correlated_confidence=True,
        )
        for i in range(cfg.random_scenarios)
    ]

    variants = {
        "full_method": TrustConfig(
            alpha=0.25,
            beta=0.25,
            gamma=0.25,
            delta=0.25,
            tau_s=cfg.tau_s,
            theta_accept=cfg.theta_accept,
            theta_reject=cfg.theta_reject,
        ),
        "no_confidence": TrustConfig(
            alpha=0.0,
            beta=1/3,
            gamma=1/3,
            delta=1/3,
            tau_s=cfg.tau_s,
            theta_accept=cfg.theta_accept,
            theta_reject=cfg.theta_reject,
        ),
        "no_noise": TrustConfig(
            alpha=1/3,
            beta=0.0,
            gamma=1/3,
            delta=1/3,
            tau_s=cfg.tau_s,
            theta_accept=cfg.theta_accept,
            theta_reject=cfg.theta_reject,
        ),
        "no_drift": TrustConfig(
            alpha=1/3,
            beta=1/3,
            gamma=0.0,
            delta=1/3,
            tau_s=cfg.tau_s,
            theta_accept=cfg.theta_accept,
            theta_reject=cfg.theta_reject,
        ),
        "no_freshness": TrustConfig(
            alpha=1/3,
            beta=1/3,
            gamma=1/3,
            delta=0.0,
            tau_s=cfg.tau_s,
            theta_accept=cfg.theta_accept,
            theta_reject=cfg.theta_reject,
        ),
    }

    detail_rows: list[dict] = []
    summary_rows: list[dict] = []

    for variant_name, trust_config in variants.items():
        trust_manager = TrustManager(trust_config)
        variant_rows: list[dict] = []

        for scenario in scenarios:
            invocation = make_invocation(scenario)

            telemetry = {
                "noise_score": scenario.noise,
                "drift_score": scenario.drift,
                "age_of_information_ms": scenario.delay_s * 1000.0,
                "output_modality": "synthetic_numeric",
            }

            result = trust_manager.evaluate_invocation(
                invocation=invocation,
                telemetry_after=telemetry,
            )

            decision = result.decision.value
            accepted = decision == "accept"
            rejected = decision == "reject"
            forwarded = decision == "forward_to_fog"

            row = {
                "variant": variant_name,
                "scenario_id": scenario.scenario_id,
                "confidence": scenario.confidence,
                "noise": scenario.noise,
                "drift": scenario.drift,
                "delay_s": scenario.delay_s,
                "freshness": result.evidence.freshness,
                "trust": result.evidence.trust,
                "true_value": scenario.true_value,
                "predicted_value": scenario.predicted_value,
                "absolute_error": scenario.absolute_error,
                "bad_output": scenario.bad_output,
                "decision": decision,
                "accepted": accepted,
                "rejected": rejected,
                "forwarded": forwarded,
                "false_acceptance": accepted and scenario.bad_output,
                "false_rejection": rejected and not scenario.bad_output,
            }

            detail_rows.append(row)
            variant_rows.append(row)

        summary = summarize_decisions(variant_rows)
        summary["variant"] = variant_name
        summary_rows.append(summary)

    save_csv(RESULTS_DIR / "ablation_details.csv", detail_rows)
    save_csv(RESULTS_DIR / "ablation_summary.csv", summary_rows)

    print("Saved ablation results.")


if __name__ == "__main__":
    main()