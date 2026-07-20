from __future__ import annotations

import csv
from pathlib import Path
from typing import Any


RESULTS_DIR = Path("evaluation/results")
INPUT_FILE = RESULTS_DIR / "trust_baselines_details.csv"
OUTPUT_FILE = RESULTS_DIR / "risk_coverage_results.csv"

# Use the main freshness configuration used in the paper.
DEFAULT_TAU_S = 5.0

# Keep the rejection threshold fixed and vary only the acceptance threshold.
THETA_REJECT = 0.35

ACCEPTANCE_THRESHOLDS = [
    0.50,
    0.55,
    0.60,
    0.65,
    0.70,
    0.75,
    0.80,
    0.85,
    0.90,
    0.95,
]


def _to_float(value: Any) -> float:
    """Convert a CSV value to float with a clear error message."""
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Cannot convert {value!r} to float.") from exc


def _to_bool(value: Any) -> bool:
    """Convert common CSV boolean representations to bool."""
    if isinstance(value, bool):
        return value

    normalized = str(value).strip().lower()

    if normalized in {"true", "1", "yes"}:
        return True

    if normalized in {"false", "0", "no"}:
        return False

    raise ValueError(f"Cannot convert {value!r} to bool.")


def load_baseline_scenarios() -> list[dict[str, Any]]:
    """
    Load one copy of each scenario from trust_baselines_details.csv.

    The baseline evaluation stores the same physical scenario once for
    each compared method. We select rows belonging to the full trust
    method at tau=5 s so that each underlying scenario is used only once.

    The selected rows still contain both:
        - confidence
        - full_trust_score

    Therefore, confidence-only and full-trust policies can be evaluated
    on exactly the same scenarios.
    """
    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"{INPUT_FILE} does not exist. "
            "Run evaluate_trust_baselines.py before this evaluation."
        )

    scenarios: list[dict[str, Any]] = []

    with INPUT_FILE.open("r", newline="") as file:
        reader = csv.DictReader(file)

        required_columns = {
            "method",
            "tau_s",
            "confidence",
            "full_trust_score",
            "absolute_error",
            "bad_output",
        }

        missing_columns = required_columns - set(reader.fieldnames or [])

        if missing_columns:
            raise ValueError(
                "trust_baselines_details.csv is missing required columns: "
                + ", ".join(sorted(missing_columns))
            )

        for row in reader:
            if row["method"] != "full_trust_score":
                continue

            if abs(_to_float(row["tau_s"]) - DEFAULT_TAU_S) > 1e-9:
                continue

            scenarios.append(
                {
                    "scenario_id": row.get("scenario_id", ""),
                    "confidence": _to_float(row["confidence"]),
                    "full_trust_score": _to_float(row["full_trust_score"]),
                    "absolute_error": _to_float(row["absolute_error"]),
                    "bad_output": _to_bool(row["bad_output"]),
                }
            )

    if not scenarios:
        raise ValueError(
            f"No full_trust_score scenarios found for tau={DEFAULT_TAU_S}."
        )

    return scenarios


def decide(
    score: float,
    theta_accept: float,
    theta_reject: float,
) -> str:
    """Apply the three-way edge decision policy."""
    if score >= theta_accept:
        return "accept"

    if score < theta_reject:
        return "reject"

    return "forward_to_fog"


def evaluate_method(
    scenarios: list[dict[str, Any]],
    method: str,
    score_field: str,
    theta_accept: float,
) -> dict[str, Any]:
    """
    Evaluate one selective decision method at one acceptance threshold.
    """
    total = len(scenarios)

    accepted = 0
    rejected = 0
    forwarded = 0

    bad_outputs = 0
    accepted_bad = 0
    accepted_good = 0

    accepted_errors: list[float] = []

    for scenario in scenarios:
        score = float(scenario[score_field])
        bad_output = bool(scenario["bad_output"])

        if bad_output:
            bad_outputs += 1

        decision = decide(
            score=score,
            theta_accept=theta_accept,
            theta_reject=THETA_REJECT,
        )

        if decision == "accept":
            accepted += 1
            accepted_errors.append(float(scenario["absolute_error"]))

            if bad_output:
                accepted_bad += 1
            else:
                accepted_good += 1

        elif decision == "reject":
            rejected += 1

        else:
            forwarded += 1

    coverage = accepted / total

    accepted_mae = (
        sum(accepted_errors) / len(accepted_errors)
        if accepted_errors
        else None
    )

    # P(accept AND bad)
    unsafe_acceptance_frequency = accepted_bad / total

    # P(accept | bad)
    false_acceptance_rate = (
        accepted_bad / bad_outputs
        if bad_outputs
        else 0.0
    )

    # P(bad | accept)
    accepted_output_contamination_rate = (
        accepted_bad / accepted
        if accepted
        else None
    )

    return {
        "method": method,
        "tau_s": DEFAULT_TAU_S,
        "theta_accept": theta_accept,
        "theta_reject": THETA_REJECT,
        "total_scenarios": total,
        "accepted": accepted,
        "forwarded": forwarded,
        "rejected": rejected,
        "coverage": coverage,
        "forwarding_rate": forwarded / total,
        "rejection_rate": rejected / total,
        "accepted_mae": accepted_mae,
        "false_acceptance_rate": false_acceptance_rate,
        "unsafe_acceptance_frequency": unsafe_acceptance_frequency,
        "accepted_output_contamination_rate": (
            accepted_output_contamination_rate
        ),
    }


def evaluate_raw_baseline(
    scenarios: list[dict[str, Any]],
) -> dict[str, Any]:
    """
    Raw phys-MCP accepts every output.

    This produces a single reference point at coverage = 1.
    """
    total = len(scenarios)

    bad_outputs = sum(
        1 for scenario in scenarios
        if bool(scenario["bad_output"])
    )

    overall_mae = sum(
        float(scenario["absolute_error"])
        for scenario in scenarios
    ) / total

    bad_rate = bad_outputs / total

    return {
        "method": "raw_phys_mcp_output",
        "tau_s": DEFAULT_TAU_S,
        "theta_accept": None,
        "theta_reject": None,
        "total_scenarios": total,
        "accepted": total,
        "forwarded": 0,
        "rejected": 0,
        "coverage": 1.0,
        "forwarding_rate": 0.0,
        "rejection_rate": 0.0,
        "accepted_mae": overall_mae,
        "false_acceptance_rate": 1.0 if bad_outputs else 0.0,
        "unsafe_acceptance_frequency": bad_rate,
        "accepted_output_contamination_rate": bad_rate,
    }


def evaluate() -> list[dict[str, Any]]:
    """
    Generate risk--coverage results.

    Confidence-only and full-trust methods are evaluated on exactly
    the same underlying physical scenarios.
    """
    scenarios = load_baseline_scenarios()

    rows: list[dict[str, Any]] = []

    # Raw phys-MCP is a fixed reference point with 100% coverage.
    rows.append(evaluate_raw_baseline(scenarios))

    for theta_accept in ACCEPTANCE_THRESHOLDS:

        rows.append(
            evaluate_method(
                scenarios=scenarios,
                method="confidence_only",
                score_field="confidence",
                theta_accept=theta_accept,
            )
        )

        rows.append(
            evaluate_method(
                scenarios=scenarios,
                method="full_trust_score",
                score_field="full_trust_score",
                theta_accept=theta_accept,
            )
        )

    return rows


def save_csv(
    path: Path,
    rows: list[dict[str, Any]],
) -> None:
    """Save result rows to CSV."""
    if not rows:
        raise ValueError("No risk-coverage results to save.")

    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(rows[0].keys()),
        )

        writer.writeheader()
        writer.writerows(rows)


def print_summary(rows: list[dict[str, Any]]) -> None:
    """Print the main risk--coverage results."""
    print("\nRisk--Coverage Analysis")
    print("=" * 80)

    print(
        f"{'Method':<25}"
        f"{'Threshold':>12}"
        f"{'Coverage':>12}"
        f"{'MAE':>12}"
        f"{'FAR':>12}"
    )

    print("-" * 80)

    for row in rows:
        threshold = row["theta_accept"]

        threshold_text = (
            "-"
            if threshold is None
            else f"{float(threshold):.2f}"
        )

        mae = row["accepted_mae"]

        mae_text = (
            "N/A"
            if mae is None
            else f"{float(mae):.6f}"
        )

        print(
            f"{row['method']:<25}"
            f"{threshold_text:>12}"
            f"{float(row['coverage']):>12.4f}"
            f"{mae_text:>12}"
            f"{float(row['false_acceptance_rate']):>12.4f}"
        )


def main() -> None:
    rows = evaluate()

    save_csv(
        OUTPUT_FILE,
        rows,
    )

    print_summary(rows)

    print(
        f"\nSaved risk--coverage results to: "
        f"{OUTPUT_FILE}"
    )


if __name__ == "__main__":
    main()