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
    """
    We do not change phys-MCP itself here.
    We create a copy-like object that has the same fields needed by TrustManager,
    but with controlled output value, confidence, and delay.
    """

    return SimpleNamespace(
        backend_id=getattr(original_invocation, "backend_id", "chemical-backend"),
        task_id=getattr(original_invocation, "task_id", "trust-eval-task"),
        output_payload={
            "value": value,
            "modality": "chemical_concentration",
        },
        confidence=confidence,
        execution_latency_ms=delay_s * 1000.0,
        backend_state=getattr(original_invocation, "backend_state", "ready"),
        notes=getattr(original_invocation, "notes", None),
    )


def evaluate() -> list[dict]:
    random.seed(42)

    orchestrator = build_default_orchestrator()
    task = make_chemical_task(task_id="trust-eval-chemical", input_level=1.8)

    base_run = orchestrator.execute_task(task)

    if not base_run.success or base_run.invocation is None:
        raise RuntimeError(f"phys-MCP invocation failed: {base_run.failure_reason}")

    true_value = 1.0
    error_threshold = 0.30

    theta_accept = 0.75
    theta_reject = 0.35

    tau_values = [1.0, 5.0, 10.0, 30.0]
    noise_values = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
    drift_values = [0.0, 0.1, 0.2, 0.3, 0.4]
    delay_values = [0.1, 1.0, 3.0, 5.0, 10.0, 20.0]

    rows: list[dict] = []

    for tau_s in tau_values:
        trust_manager = TrustManager(
            TrustConfig(
                alpha=0.25,
                beta=0.25,
                gamma=0.25,
                delta=0.25,
                tau_s=tau_s,
                theta_accept=theta_accept,
                theta_reject=theta_reject,
            )
        )

        for noise_level in noise_values:
            for drift_level in drift_values:
                for delay_s in delay_values:
                    for repetition in range(30):
                        random_noise = random.gauss(0.0, noise_level)
                        predicted_value = true_value + random_noise + drift_level
                        absolute_error = abs(predicted_value - true_value)

                        bad_output = absolute_error > error_threshold

                        confidence = max(0.0, min(1.0, 1.0 - noise_level - drift_level))

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
                                "method_score": None,  # raw baseline has no trust score
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
                                    "method": method_name,
                                    "tau_s": tau_s,
                                    "noise_level": noise_level,
                                    "drift_level": drift_level,
                                    "delay_s": delay_s,
                                    "repetition": repetition,
                                    "true_value": true_value,
                                    "predicted_value": predicted_value,
                                    "absolute_error": absolute_error,
                                    "bad_output": bad_output,
                                    "confidence": confidence,
                                    "freshness": freshness,
                                    "full_trust_score": full_trust_score,
                                    "method_score": method_score,
                                    "decision": decision,
                                    "accepted": accepted,
                                    "rejected": rejected,
                                    "forwarded": forwarded,
                                    "false_acceptance": false_acceptance,
                                    "false_rejection": false_rejection,
                                }
                            )
    return rows









def _mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)


def _mean_or_none(values):
    if not values:
        return None
    return sum(values) / len(values)


def summarize(rows: list[dict]) -> list[dict]:
    groups: dict[tuple, list[dict]] = {}

    for row in rows:
        key = (
            row["method"],
            row["tau_s"],
            row["noise_level"],
            row["drift_level"],
            row["delay_s"],
        )
        groups.setdefault(key, []).append(row)

    summary_rows: list[dict] = []

    for key, items in groups.items():
        method, tau_s, noise_level, drift_level, delay_s = key
        n = len(items)

        bad_items = [x for x in items if bool(x["bad_output"])]
        good_items = [x for x in items if not bool(x["bad_output"])]
        accepted_items = [x for x in items if bool(x["accepted"])]
        rejected_items = [x for x in items if bool(x["rejected"])]
        forwarded_items = [x for x in items if bool(x["forwarded"])]

        accepted_bad_items = [
            x for x in items
            if bool(x["accepted"]) and bool(x["bad_output"])
        ]

        rejected_good_items = [
            x for x in items
            if bool(x["rejected"]) and not bool(x["bad_output"])
        ]

        conditional_false_acceptance_rate = (
            len(accepted_bad_items) / len(bad_items)
            if bad_items else 0.0
        )

        conditional_false_rejection_rate = (
            len(rejected_good_items) / len(good_items)
            if good_items else 0.0
        )

        bad_rate_among_accepted = (
            len(accepted_bad_items) / len(accepted_items)
            if accepted_items else None
        )

        overall_mae = _mean_or_none([
            float(x["absolute_error"])
            for x in items
        ])

        accepted_mae = _mean_or_none([
            float(x["absolute_error"])
            for x in accepted_items
        ])

        mean_full_trust_score = _mean_or_none([
            float(x["full_trust_score"])
            for x in items
        ])

        method_scores = [
            float(x["method_score"])
            for x in items
            if x["method_score"] is not None
        ]

        mean_method_score = _mean_or_none(method_scores)

        summary_rows.append(
            {
                "method": method,
                "tau_s": tau_s,
                "noise_level": noise_level,
                "drift_level": drift_level,
                "delay_s": delay_s,
                "runs": n,

                "overall_mae": overall_mae,

                # This is empty/NaN if the method accepted no outputs.
                "accepted_mae": accepted_mae,

                "bad_output_rate": len(bad_items) / n,
                "acceptance_rate": len(accepted_items) / n,
                "rejection_rate": len(rejected_items) / n,
                "forwarding_rate": len(forwarded_items) / n,

                "false_acceptance_rate_all_runs": len(accepted_bad_items) / n,
                "false_rejection_rate_all_runs": len(rejected_good_items) / n,

                "conditional_false_acceptance_rate": conditional_false_acceptance_rate,
                "conditional_false_rejection_rate": conditional_false_rejection_rate,

                # This is empty/NaN if the method accepted no outputs.
                "bad_rate_among_accepted": bad_rate_among_accepted,

                # Full trust score is still saved for analysis.
                "mean_full_trust_score": mean_full_trust_score,

                # This is the actual score used by each method.
                # raw_phys_mcp_output has no score, so it is empty/NaN.
                "mean_method_score": mean_method_score,
            }
        )

    return summary_rows



def main() -> None:
    rows = evaluate()
    summary_rows = summarize(rows)

    details_path = RESULTS_DIR / "trust_baselines_details.csv"
    summary_path = RESULTS_DIR / "trust_baselines_summary.csv"

    save_csv(details_path, rows)
    save_csv(summary_path, summary_rows)

    print(f"Saved details to: {details_path}")
    print(f"Saved summary to: {summary_path}")


if __name__ == "__main__":
    main()