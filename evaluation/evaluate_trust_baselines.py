from __future__ import annotations

import csv
import random
from pathlib import Path
from types import SimpleNamespace

from core.trust_manager import (
    TrustConfig,
    TrustManager,
)

from demos.common import (
    build_default_orchestrator,
    make_chemical_task,
)

from evaluation.scenario_generator import (
    make_controlled_scenario,
)


# ---------------------------------------------------------------------
# Results directory
# ---------------------------------------------------------------------

RESULTS_DIR = Path(
    "evaluation/results"
)

RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ---------------------------------------------------------------------
# CSV utility
# ---------------------------------------------------------------------

def save_csv(
    path: Path,
    rows: list[dict],
) -> None:
    """
    Save experiment results to a CSV file.
    """

    if not rows:
        raise ValueError(
            "No rows to save."
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

        writer.writerows(
            rows
        )


# ---------------------------------------------------------------------
# Three-way decision policy
# ---------------------------------------------------------------------

def decide_from_score(
    score: float,
    theta_accept: float,
    theta_reject: float,
) -> str:
    """
    Apply the same three-way decision policy used by TrustManager.

    High score:
        Accept locally.

    Low score:
        Reject.

    Intermediate score:
        Forward to fog.
    """

    if score >= theta_accept:
        return "accept"

    if score < theta_reject:
        return "reject"

    return "forward_to_fog"


# ---------------------------------------------------------------------
# Modified invocation construction
# ---------------------------------------------------------------------

def make_modified_invocation(
    original_invocation,
    value: float,
    confidence: float,
    delay_s: float,
):
    """
    Create a copy-like invocation object containing the fields needed
    by TrustManager.

    The original phys-MCP implementation itself is not modified.
    """

    return SimpleNamespace(

        backend_id=getattr(
            original_invocation,
            "backend_id",
            "chemical-backend",
        ),

        task_id=getattr(
            original_invocation,
            "task_id",
            "trust-eval-task",
        ),

        output_payload={
            "value":
                value,

            "modality":
                "chemical_concentration",
        },

        confidence=(
            confidence
        ),

        execution_latency_ms=(
            delay_s
            * 1000.0
        ),

        backend_state=getattr(
            original_invocation,
            "backend_state",
            "ready",
        ),

        notes=getattr(
            original_invocation,
            "notes",
            None,
        ),
    )


# ---------------------------------------------------------------------
# Main trust baseline evaluation
# ---------------------------------------------------------------------

def evaluate(
    seed: int = 42,
) -> list[dict]:
    """
    Evaluate trust-aware output handling under controlled physical
    uncertainty.

    The experiment varies:

        - true physical noise,
        - true physical drift,
        - output delay,
        - freshness time constant tau_s.

    Important design properties:

    1. Ground-truth noise and drift are used only to generate the
       physical output.

    2. TrustManager receives imperfect estimated noise and drift,
       rather than the exact ground-truth conditions.

    3. Severe noise and drift conditions are included so that all
       three decisions can naturally occur:

           ACCEPT
           FORWARD_TO_FOG
           REJECT

    4. The exact same generated scenarios are reused for every tau_s
       configuration. Therefore, differences between tau settings
       result only from freshness decay and not from different random
       physical outputs.

    5. The random seed can be externally specified so that the same
       experiment can be repeated across multiple independent seeds.
    """

    # -----------------------------------------------------------------
    # Reproducibility
    #
    # Default seed=42 preserves the behavior of the original
    # single-seed experiment.
    #
    # Multi-seed evaluation can call:
    #
    #     evaluate(seed=101)
    #     evaluate(seed=202)
    #     ...
    # -----------------------------------------------------------------

    random.seed(
        seed
    )

    # -----------------------------------------------------------------
    # Build base phys-MCP invocation
    # -----------------------------------------------------------------

    orchestrator = (
        build_default_orchestrator()
    )

    task = make_chemical_task(
        task_id=(
            "trust-eval-chemical"
        ),
        input_level=1.8,
    )

    base_run = (
        orchestrator.execute_task(
            task
        )
    )

    if (
        not base_run.success
        or base_run.invocation is None
    ):

        raise RuntimeError(
            "phys-MCP invocation failed: "
            f"{base_run.failure_reason}"
        )

    # -----------------------------------------------------------------
    # General experiment configuration
    # -----------------------------------------------------------------

    true_value = (
        1.0
    )

    error_threshold = (
        0.30
    )

    theta_accept = (
        0.75
    )

    theta_reject = (
        0.35
    )

    # -----------------------------------------------------------------
    # Freshness sensitivity configurations
    # -----------------------------------------------------------------

    tau_values = [
        1.0,
        5.0,
        10.0,
        30.0,
    ]

    # -----------------------------------------------------------------
    # TRUE controlled physical noise levels
    #
    # Severe values 0.7 and 0.9 are included so that highly unreliable
    # outputs can naturally enter the rejection region.
    # -----------------------------------------------------------------

    noise_values = [
        0.0,
        0.1,
        0.2,
        0.3,
        0.4,
        0.5,
        0.7,
        0.9,
    ]

    # -----------------------------------------------------------------
    # TRUE controlled physical drift levels
    #
    # Severe values 0.6 and 0.8 represent strong substrate degradation.
    # -----------------------------------------------------------------

    drift_values = [
        0.0,
        0.1,
        0.2,
        0.3,
        0.4,
        0.6,
        0.8,
    ]

    # -----------------------------------------------------------------
    # Output delay values
    # -----------------------------------------------------------------

    delay_values = [
        0.1,
        1.0,
        3.0,
        5.0,
        10.0,
        20.0,
    ]

    # -----------------------------------------------------------------
    # Repeated runs per condition
    # -----------------------------------------------------------------

    repetitions = (
        30
    )

    rows: list[
        dict
    ] = []

    # =================================================================
    # STEP 1:
    # Generate one COMMON scenario set.
    #
    # This happens before the tau loop.
    #
    # The same:
    #
    #     true noise,
    #     true drift,
    #     estimated noise,
    #     estimated drift,
    #     confidence,
    #     physical output,
    #     delay,
    #
    # will therefore be reused for every tau_s value.
    # =================================================================

    common_scenarios: list[
        dict
    ] = []

    scenario_id = (
        0
    )

    for true_noise in noise_values:

        for true_drift in drift_values:

            for delay_s in delay_values:

                for repetition in range(
                    repetitions
                ):

                    scenario = (
                        make_controlled_scenario(

                            scenario_id=(
                                scenario_id
                            ),

                            true_value=(
                                true_value
                            ),

                            noise=(
                                true_noise
                            ),

                            drift=(
                                true_drift
                            ),

                            delay_s=(
                                delay_s
                            ),

                            error_threshold=(
                                error_threshold
                            ),
                        )
                    )

                    common_scenarios.append(
                        {
                            "scenario":
                                scenario,

                            "repetition":
                                repetition,
                        }
                    )

                    scenario_id += (
                        1
                    )

    print(
        "Generated common controlled "
        f"scenario set for seed {seed}: "
        f"{len(common_scenarios)} scenarios"
    )

    # =================================================================
    # STEP 2:
    # Evaluate the SAME scenarios for each tau_s value.
    # =================================================================

    for tau_s in tau_values:

        # -------------------------------------------------------------
        # Only freshness time constant changes here.
        # -------------------------------------------------------------

        trust_manager = (
            TrustManager(

                TrustConfig(

                    alpha=0.25,

                    beta=0.25,

                    gamma=0.25,

                    delta=0.25,

                    tau_s=(
                        tau_s
                    ),

                    theta_accept=(
                        theta_accept
                    ),

                    theta_reject=(
                        theta_reject
                    ),
                )
            )
        )

        # -------------------------------------------------------------
        # Reuse exact same scenario set.
        # -------------------------------------------------------------

        for scenario_entry in (
            common_scenarios
        ):

            scenario = (
                scenario_entry[
                    "scenario"
                ]
            )

            repetition = (
                scenario_entry[
                    "repetition"
                ]
            )

            # =========================================================
            # Build PNN invocation
            #
            # predicted_value and confidence remain identical across
            # different tau configurations.
            # =========================================================

            invocation = (
                make_modified_invocation(

                    original_invocation=(
                        base_run.invocation
                    ),

                    value=(
                        scenario
                        .predicted_value
                    ),

                    confidence=(
                        scenario
                        .confidence
                    ),

                    delay_s=(
                        scenario
                        .delay_s
                    ),
                )
            )

            # =========================================================
            # Construct telemetry visible to TrustManager
            #
            # IMPORTANT:
            #
            # scenario.noise
            # scenario.drift
            #
            # are imperfect estimates.
            #
            # The hidden values:
            #
            # scenario.true_noise
            # scenario.true_drift
            #
            # are NOT passed to TrustManager.
            # =========================================================

            telemetry = dict(
                base_run.telemetry_after
            )

            telemetry.update(
                {
                    "noise_score":
                        scenario.noise,

                    "drift_score":
                        scenario.drift,

                    "age_of_information_ms":
                        (
                            scenario.delay_s
                            * 1000.0
                        ),

                    "output_modality":
                        "chemical_concentration",
                }
            )

            # =========================================================
            # Evaluate proposed trust method
            # =========================================================

            trust_result = (
                trust_manager
                .evaluate_invocation(

                    invocation=(
                        invocation
                    ),

                    telemetry_after=(
                        telemetry
                    ),
                )
            )

            freshness = (
                trust_result
                .evidence
                .freshness
            )

            full_trust_score = (
                trust_result
                .evidence
                .trust
            )

            # =========================================================
            # Evaluation methods
            # =========================================================

            methods = {

                # -----------------------------------------------------
                # Baseline 1:
                # Raw phys-MCP output
                #
                # No post-invocation reliability filtering.
                # -----------------------------------------------------

                "raw_phys_mcp_output": {

                    "decision":
                        "accept",

                    "method_score":
                        None,
                },

                # -----------------------------------------------------
                # Baseline 2:
                # Confidence-only decision
                # -----------------------------------------------------

                "confidence_only": {

                    "decision":
                        decide_from_score(

                            scenario.confidence,

                            theta_accept,

                            theta_reject,
                        ),

                    "method_score":
                        scenario.confidence,
                },

                # -----------------------------------------------------
                # Baseline 3:
                # Freshness-only decision
                # -----------------------------------------------------

                "freshness_only": {

                    "decision":
                        decide_from_score(

                            freshness,

                            theta_accept,

                            theta_reject,
                        ),

                    "method_score":
                        freshness,
                },

                # -----------------------------------------------------
                # Proposed method:
                # Full multi-factor trust score
                # -----------------------------------------------------

                "full_trust_score": {

                    "decision":
                        trust_result
                        .decision
                        .value,

                    "method_score":
                        full_trust_score,
                },
            }

            # =========================================================
            # Store one result per evaluation method
            # =========================================================

            for (
                method_name,
                method_info,
            ) in methods.items():

                decision = (
                    method_info[
                        "decision"
                    ]
                )

                method_score = (
                    method_info[
                        "method_score"
                    ]
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

                rows.append(
                    {
                        # ---------------------------------------------
                        # Random seed
                        #
                        # Including the seed in the detailed result
                        # makes multi-seed results traceable.
                        # ---------------------------------------------

                        "seed":
                            seed,

                        # ---------------------------------------------
                        # Method
                        # ---------------------------------------------

                        "method":
                            method_name,

                        # ---------------------------------------------
                        # Freshness configuration
                        # ---------------------------------------------

                        "tau_s":
                            tau_s,

                        # ---------------------------------------------
                        # Scenario identification
                        #
                        # Same scenario_id appears for every tau value.
                        # ---------------------------------------------

                        "scenario_id":
                            scenario.scenario_id,

                        "repetition":
                            repetition,

                        # ---------------------------------------------
                        # TRUE controlled physical conditions
                        # ---------------------------------------------

                        "noise_level":
                            scenario.true_noise,

                        "drift_level":
                            scenario.true_drift,

                        "true_noise":
                            scenario.true_noise,

                        "true_drift":
                            scenario.true_drift,

                        # ---------------------------------------------
                        # Imperfect telemetry visible to TrustManager
                        # ---------------------------------------------

                        "estimated_noise":
                            scenario.noise,

                        "estimated_drift":
                            scenario.drift,

                        # ---------------------------------------------
                        # Delay and freshness
                        # ---------------------------------------------

                        "delay_s":
                            scenario.delay_s,

                        "freshness":
                            freshness,

                        # ---------------------------------------------
                        # Confidence
                        # ---------------------------------------------

                        "confidence":
                            scenario.confidence,

                        # ---------------------------------------------
                        # Physical output
                        # ---------------------------------------------

                        "true_value":
                            scenario.true_value,

                        "predicted_value":
                            scenario.predicted_value,

                        "absolute_error":
                            scenario.absolute_error,

                        "bad_output":
                            scenario.bad_output,

                        # ---------------------------------------------
                        # Trust
                        # ---------------------------------------------

                        "full_trust_score":
                            full_trust_score,

                        "method_score":
                            method_score,

                        # ---------------------------------------------
                        # Decision
                        # ---------------------------------------------

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
                )

    return (
        rows
    )


# ---------------------------------------------------------------------
# Mean utility
# ---------------------------------------------------------------------

def _mean_or_none(
    values,
):
    """
    Calculate arithmetic mean.

    Return None when the input list is empty.
    """

    if not values:
        return None

    return (
        sum(
            values
        )
        / len(
            values
        )
    )


# ---------------------------------------------------------------------
# Summary generation
# ---------------------------------------------------------------------

def summarize(
    rows: list[dict],
) -> list[dict]:
    """
    Aggregate results for each combination of:

        method,
        tau_s,
        true noise,
        true drift,
        delay.
    """

    groups: dict[
        tuple,
        list[dict],
    ] = {}

    # -----------------------------------------------------------------
    # Group rows
    # -----------------------------------------------------------------

    for row in rows:

        key = (
            row[
                "method"
            ],

            row[
                "tau_s"
            ],

            row[
                "noise_level"
            ],

            row[
                "drift_level"
            ],

            row[
                "delay_s"
            ],
        )

        groups.setdefault(
            key,
            [],
        ).append(
            row
        )

    summary_rows: list[
        dict
    ] = []

    # =================================================================
    # Summarize each group
    # =================================================================

    for (
        key,
        items,
    ) in groups.items():

        (
            method,
            tau_s,
            noise_level,
            drift_level,
            delay_s,
        ) = key

        n = len(
            items
        )

        # -------------------------------------------------------------
        # Separate output quality categories
        # -------------------------------------------------------------

        bad_items = [
            item
            for item
            in items
            if bool(
                item[
                    "bad_output"
                ]
            )
        ]

        good_items = [
            item
            for item
            in items
            if not bool(
                item[
                    "bad_output"
                ]
            )
        ]

        # -------------------------------------------------------------
        # Separate decision categories
        # -------------------------------------------------------------

        accepted_items = [
            item
            for item
            in items
            if bool(
                item[
                    "accepted"
                ]
            )
        ]

        rejected_items = [
            item
            for item
            in items
            if bool(
                item[
                    "rejected"
                ]
            )
        ]

        forwarded_items = [
            item
            for item
            in items
            if bool(
                item[
                    "forwarded"
                ]
            )
        ]

        # -------------------------------------------------------------
        # Incorrect decisions
        # -------------------------------------------------------------

        accepted_bad_items = [
            item
            for item
            in items
            if (
                bool(
                    item[
                        "accepted"
                    ]
                )
                and bool(
                    item[
                        "bad_output"
                    ]
                )
            )
        ]

        rejected_good_items = [
            item
            for item
            in items
            if (
                bool(
                    item[
                        "rejected"
                    ]
                )
                and not bool(
                    item[
                        "bad_output"
                    ]
                )
            )
        ]

        # -------------------------------------------------------------
        # Conditional false acceptance rate
        #
        # Among all bad outputs:
        # how many were incorrectly accepted?
        # -------------------------------------------------------------

        conditional_false_acceptance_rate = (

            len(
                accepted_bad_items
            )
            / len(
                bad_items
            )

            if bad_items

            else 0.0
        )

        # -------------------------------------------------------------
        # Conditional false rejection rate
        #
        # Among all good outputs:
        # how many were incorrectly rejected?
        # -------------------------------------------------------------

        conditional_false_rejection_rate = (

            len(
                rejected_good_items
            )
            / len(
                good_items
            )

            if good_items

            else 0.0
        )

        # -------------------------------------------------------------
        # Fraction of accepted outputs that were bad
        # -------------------------------------------------------------

        bad_rate_among_accepted = (

            len(
                accepted_bad_items
            )
            / len(
                accepted_items
            )

            if accepted_items

            else None
        )

        # -------------------------------------------------------------
        # Overall MAE
        # -------------------------------------------------------------

        overall_mae = (
            _mean_or_none(
                [
                    float(
                        item[
                            "absolute_error"
                        ]
                    )

                    for item
                    in items
                ]
            )
        )

        # -------------------------------------------------------------
        # MAE only among accepted outputs
        # -------------------------------------------------------------

        accepted_mae = (
            _mean_or_none(
                [
                    float(
                        item[
                            "absolute_error"
                        ]
                    )

                    for item
                    in accepted_items
                ]
            )
        )

        # -------------------------------------------------------------
        # Mean full trust score
        # -------------------------------------------------------------

        mean_full_trust_score = (
            _mean_or_none(
                [
                    float(
                        item[
                            "full_trust_score"
                        ]
                    )

                    for item
                    in items
                ]
            )
        )

        # -------------------------------------------------------------
        # Method-specific score
        #
        # Raw phys-MCP has no score.
        # -------------------------------------------------------------

        method_scores = [

            float(
                item[
                    "method_score"
                ]
            )

            for item
            in items

            if (
                item[
                    "method_score"
                ]
                is not None
            )
        ]

        mean_method_score = (
            _mean_or_none(
                method_scores
            )
        )

        # -------------------------------------------------------------
        # Mean imperfect telemetry values
        # -------------------------------------------------------------

        mean_estimated_noise = (
            _mean_or_none(
                [
                    float(
                        item[
                            "estimated_noise"
                        ]
                    )

                    for item
                    in items
                ]
            )
        )

        mean_estimated_drift = (
            _mean_or_none(
                [
                    float(
                        item[
                            "estimated_drift"
                        ]
                    )

                    for item
                    in items
                ]
            )
        )

        mean_confidence = (
            _mean_or_none(
                [
                    float(
                        item[
                            "confidence"
                        ]
                    )

                    for item
                    in items
                ]
            )
        )

        # -------------------------------------------------------------
        # Store summary row
        # -------------------------------------------------------------

        summary_rows.append(
            {
                "method":
                    method,

                "tau_s":
                    tau_s,

                # -----------------------------------------
                # TRUE controlled physical conditions
                # -----------------------------------------

                "noise_level":
                    noise_level,

                "drift_level":
                    drift_level,

                "delay_s":
                    delay_s,

                "runs":
                    n,

                # -----------------------------------------
                # Accuracy
                # -----------------------------------------

                "overall_mae":
                    overall_mae,

                "accepted_mae":
                    accepted_mae,

                "bad_output_rate":
                    (
                        len(
                            bad_items
                        )
                        / n
                    ),

                # -----------------------------------------
                # Decision rates
                # -----------------------------------------

                "acceptance_rate":
                    (
                        len(
                            accepted_items
                        )
                        / n
                    ),

                "rejection_rate":
                    (
                        len(
                            rejected_items
                        )
                        / n
                    ),

                "forwarding_rate":
                    (
                        len(
                            forwarded_items
                        )
                        / n
                    ),

                # -----------------------------------------
                # Error rates over all runs
                # -----------------------------------------

                "false_acceptance_rate_all_runs":
                    (
                        len(
                            accepted_bad_items
                        )
                        / n
                    ),

                "false_rejection_rate_all_runs":
                    (
                        len(
                            rejected_good_items
                        )
                        / n
                    ),

                # -----------------------------------------
                # Conditional error rates
                # -----------------------------------------

                "conditional_false_acceptance_rate":
                    conditional_false_acceptance_rate,

                "conditional_false_rejection_rate":
                    conditional_false_rejection_rate,

                "bad_rate_among_accepted":
                    bad_rate_among_accepted,

                # -----------------------------------------
                # Trust statistics
                # -----------------------------------------

                "mean_full_trust_score":
                    mean_full_trust_score,

                "mean_method_score":
                    mean_method_score,

                # -----------------------------------------
                # Imperfect telemetry statistics
                # -----------------------------------------

                "mean_estimated_noise":
                    mean_estimated_noise,

                "mean_estimated_drift":
                    mean_estimated_drift,

                "mean_confidence":
                    mean_confidence,
            }
        )

    return (
        summary_rows
    )


# ---------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------

def main() -> None:
    """
    Run the default single-seed controlled trust-baseline experiment.

    The default run uses seed=42 to preserve the original experimental
    configuration.

    Generates:

        evaluation/results/trust_baselines_details.csv

        evaluation/results/trust_baselines_summary.csv
    """

    rows = (
        evaluate(
            seed=42
        )
    )

    summary_rows = (
        summarize(
            rows
        )
    )

    details_path = (
        RESULTS_DIR
        / "trust_baselines_details.csv"
    )

    summary_path = (
        RESULTS_DIR
        / "trust_baselines_summary.csv"
    )

    save_csv(
        details_path,
        rows,
    )

    save_csv(
        summary_path,
        summary_rows,
    )

    print()

    print(
        "Trust baseline evaluation completed."
    )

    print(
        "Random seed: 42"
    )

    print(
        f"Total detailed rows: "
        f"{len(rows)}"
    )

    print(
        f"Total summary rows: "
        f"{len(summary_rows)}"
    )

    print(
        "Saved details to: "
        f"{details_path}"
    )

    print(
        "Saved summary to: "
        f"{summary_path}"
    )


if __name__ == "__main__":
    main()