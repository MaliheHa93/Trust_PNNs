from __future__ import annotations

import csv
import random
from pathlib import Path

from core.trust_manager import TrustConfig, TrustManager
from evaluation.experiment_config import ExperimentConfig
from evaluation.metrics import summarize_decisions
from evaluation.scenario_generator import make_random_scenario


# ---------------------------------------------------------------------
# Output directory
# ---------------------------------------------------------------------

RESULTS_DIR = Path("evaluation/results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------

def save_csv(
    path: Path,
    rows: list[dict],
) -> None:
    """
    Save a list of dictionaries as a CSV file.
    """

    if not rows:
        raise ValueError(
            f"No rows available to save to {path}."
        )

    with path.open(
        "w",
        newline="",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=list(
                rows[0].keys()
            ),
        )

        writer.writeheader()
        writer.writerows(rows)


def decide_from_trust(
    trust: float,
    theta_accept: float,
    theta_reject: float,
) -> str:
    """
    Apply the three-way trust decision policy.

    High trust:
        Accept locally.

    Low trust:
        Reject.

    Intermediate trust:
        Forward to fog.
    """

    if trust >= theta_accept:
        return "accept"

    if trust < theta_reject:
        return "reject"

    return "forward_to_fog"


# ---------------------------------------------------------------------
# Trust-weight configurations
# ---------------------------------------------------------------------

WEIGHT_CONFIGS: dict[
    str,
    tuple[
        float,
        float,
        float,
        float,
    ],
] = {

    # Default substrate-agnostic configuration.
    "equal": (
        0.25,  # confidence
        0.25,  # noise
        0.25,  # drift
        0.25,  # freshness
    ),

    # Give more importance to confidence.
    "confidence_priority": (
        0.40,
        0.20,
        0.20,
        0.20,
    ),

    # Give more importance to noise.
    "noise_priority": (
        0.20,
        0.40,
        0.20,
        0.20,
    ),

    # Give more importance to drift.
    "drift_priority": (
        0.20,
        0.20,
        0.40,
        0.20,
    ),

    # Give more importance to freshness.
    "freshness_priority": (
        0.20,
        0.20,
        0.20,
        0.40,
    ),
}


# ---------------------------------------------------------------------
# Decision-threshold configurations
# ---------------------------------------------------------------------

THRESHOLD_CONFIGS: list[
    tuple[
        float,
        float,
    ]
] = [

    # More permissive decision policy.
    (
        0.70,  # theta_accept
        0.30,  # theta_reject
    ),

    # Default configuration used in the paper.
    (
        0.75,
        0.35,
    ),

    # More conservative acceptance policy.
    (
        0.80,
        0.40,
    ),

    # Strongly conservative acceptance policy.
    (
        0.85,
        0.45,
    ),
]


# ---------------------------------------------------------------------
# Sensitivity evaluation
# ---------------------------------------------------------------------

def evaluate() -> tuple[
    list[dict],
    list[dict],
]:
    """
    Evaluate sensitivity to:

    1. Trust-component weights:
       alpha, beta, gamma, delta

    2. Decision thresholds:
       theta_accept, theta_reject

    The same randomly generated scenarios are reused for every
    configuration so that differences in results arise from the
    trust configuration rather than different random samples.
    """

    # -------------------------------------------------------------
    # Load the shared experimental configuration
    # -------------------------------------------------------------

    cfg = ExperimentConfig()

    # The scenario generator currently uses Python's global random
    # generator, so setting the seed here ensures reproducibility.
    random.seed(
        cfg.seed
    )

    # -------------------------------------------------------------
    # Generate ONE common scenario set
    #
    # IMPORTANT:
    # These exact same scenarios are used for every weight and
    # threshold configuration.
    #
    # This provides a fair comparison.
    # -------------------------------------------------------------

    scenarios = [

        make_random_scenario(
            scenario_id=i,
            true_value=cfg.true_value,
            error_threshold=cfg.error_threshold,
            correlated_confidence=True,
        )

        for i in range(
            cfg.random_scenarios
        )
    ]

    print(
        f"Generated "
        f"{len(scenarios)} "
        f"common scenarios."
    )

    detail_rows: list[dict] = []
    summary_rows: list[dict] = []

    # -------------------------------------------------------------
    # Evaluate every weight configuration
    # -------------------------------------------------------------

    for (
        weight_name,
        weights,
    ) in WEIGHT_CONFIGS.items():

        (
            alpha,
            beta,
            gamma,
            delta,
        ) = weights

        # ---------------------------------------------------------
        # Evaluate every threshold configuration
        # ---------------------------------------------------------

        for (
            theta_accept,
            theta_reject,
        ) in THRESHOLD_CONFIGS:

            # -----------------------------------------------------
            # Create TrustManager for this configuration
            # -----------------------------------------------------

            trust_manager = TrustManager(

                TrustConfig(
                    alpha=alpha,
                    beta=beta,
                    gamma=gamma,
                    delta=delta,

                    tau_s=cfg.tau_s,

                    theta_accept=(
                        theta_accept
                    ),

                    theta_reject=(
                        theta_reject
                    ),
                )
            )

            configuration_rows: list[
                dict
            ] = []

            # -----------------------------------------------------
            # Evaluate the common scenarios
            # -----------------------------------------------------

            for scenario in scenarios:

                # -------------------------------------------------
                # Compute freshness
                # -------------------------------------------------

                freshness = (
                    trust_manager.compute_freshness(
                        scenario.delay_s
                    )
                )

                # -------------------------------------------------
                # Compute trust
                #
                # scenario.noise and scenario.drift are the
                # IMPERFECT TELEMETRY ESTIMATES generated by the
                # updated scenario_generator.py.
                #
                # scenario.true_noise and scenario.true_drift are
                # hidden physical conditions and are NOT used by
                # TrustManager.
                # -------------------------------------------------

                trust = (
                    trust_manager.compute_trust(

                        confidence=(
                            scenario.confidence
                        ),

                        noise=(
                            scenario.noise
                        ),

                        drift=(
                            scenario.drift
                        ),

                        freshness=(
                            freshness
                        ),
                    )
                )

                # -------------------------------------------------
                # Apply decision policy
                # -------------------------------------------------

                decision = decide_from_trust(

                    trust=trust,

                    theta_accept=(
                        theta_accept
                    ),

                    theta_reject=(
                        theta_reject
                    ),
                )

                accepted = (
                    decision
                    == "accept"
                )

                rejected = (
                    decision
                    == "reject"
                )

                forwarded = (
                    decision
                    == "forward_to_fog"
                )

                false_acceptance = (
                    accepted
                    and scenario.bad_output
                )

                false_rejection = (
                    rejected
                    and not scenario.bad_output
                )

                # -------------------------------------------------
                # Store detailed result
                # -------------------------------------------------

                row = {

                    # Configuration
                    "weight_config":
                        weight_name,

                    "alpha":
                        alpha,

                    "beta":
                        beta,

                    "gamma":
                        gamma,

                    "delta":
                        delta,

                    "theta_accept":
                        theta_accept,

                    "theta_reject":
                        theta_reject,

                    "tau_s":
                        cfg.tau_s,

                    # Scenario identification
                    "scenario_id":
                        scenario.scenario_id,

                    # Ground-truth output
                    "true_value":
                        scenario.true_value,

                    "predicted_value":
                        scenario.predicted_value,

                    "absolute_error":
                        scenario.absolute_error,

                    "bad_output":
                        scenario.bad_output,

                    # Hidden physical conditions
                    #
                    # These are stored for analysis only.
                    # They are NOT passed to TrustManager.
                    "true_noise":
                        scenario.true_noise,

                    "true_drift":
                        scenario.true_drift,

                    # Imperfect telemetry visible
                    # to TrustManager
                    "estimated_noise":
                        scenario.noise,

                    "estimated_drift":
                        scenario.drift,

                    "confidence":
                        scenario.confidence,

                    "delay_s":
                        scenario.delay_s,

                    "freshness":
                        freshness,

                    # Trust result
                    "trust":
                        trust,

                    "decision":
                        decision,

                    "accepted":
                        accepted,

                    "rejected":
                        rejected,

                    "forwarded":
                        forwarded,

                    "false_acceptance":
                        false_acceptance,

                    "false_rejection":
                        false_rejection,
                }

                detail_rows.append(
                    row
                )

                configuration_rows.append(
                    row
                )

            # -----------------------------------------------------
            # Summarize this configuration
            #
            # summarize_decisions() expects:
            #
            # absolute_error
            # bad_output
            # accepted
            # rejected
            # forwarded
            #
            # All are included above.
            # -----------------------------------------------------

            summary = (
                summarize_decisions(
                    configuration_rows
                )
            )

            # -----------------------------------------------------
            # Add configuration information to summary
            # -----------------------------------------------------

            summary[
                "weight_config"
            ] = weight_name

            summary[
                "alpha"
            ] = alpha

            summary[
                "beta"
            ] = beta

            summary[
                "gamma"
            ] = gamma

            summary[
                "delta"
            ] = delta

            summary[
                "theta_accept"
            ] = theta_accept

            summary[
                "theta_reject"
            ] = theta_reject

            summary[
                "tau_s"
            ] = cfg.tau_s

            # -----------------------------------------------------
            # Additional useful summary statistics
            # -----------------------------------------------------

            mean_trust = (

                sum(
                    float(
                        row["trust"]
                    )
                    for row
                    in configuration_rows
                )

                / len(
                    configuration_rows
                )
            )

            mean_confidence = (

                sum(
                    float(
                        row["confidence"]
                    )
                    for row
                    in configuration_rows
                )

                / len(
                    configuration_rows
                )
            )

            summary[
                "mean_trust"
            ] = mean_trust

            summary[
                "mean_confidence"
            ] = mean_confidence

            summary_rows.append(
                summary
            )

            # -----------------------------------------------------
            # Console progress
            # -----------------------------------------------------

            print(

                f"Completed: "
                f"{weight_name} | "

                f"weights="
                f"({alpha:.2f}, "
                f"{beta:.2f}, "
                f"{gamma:.2f}, "
                f"{delta:.2f}) | "

                f"accept="
                f"{theta_accept:.2f} | "

                f"reject="
                f"{theta_reject:.2f}"
            )

    return (
        detail_rows,
        summary_rows,
    )


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main() -> None:
    """
    Run the complete sensitivity analysis and save:

    1. Detailed per-scenario results.
    2. Aggregated results for each weight/threshold configuration.
    """

    (
        detail_rows,
        summary_rows,
    ) = evaluate()

    details_path = (
        RESULTS_DIR
        / "sensitivity_details.csv"
    )

    summary_path = (
        RESULTS_DIR
        / "sensitivity_summary.csv"
    )

    save_csv(
        details_path,
        detail_rows,
    )

    save_csv(
        summary_path,
        summary_rows,
    )

    print()
    print(
        "Sensitivity evaluation completed."
    )

    print(
        f"Saved detailed results to: "
        f"{details_path}"
    )

    print(
        f"Saved summary results to: "
        f"{summary_path}"
    )


if __name__ == "__main__":
    main()