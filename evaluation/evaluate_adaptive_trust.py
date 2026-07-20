from __future__ import annotations

import csv
import random
from pathlib import Path

from core.adaptive_trust import (
    AdaptiveTrustConfig,
    AdaptiveTrustTracker,
)
from core.trust_manager import (
    TrustConfig,
    TrustManager,
)


# ---------------------------------------------------------------------
# Output directory
# ---------------------------------------------------------------------

RESULTS_DIR = Path(
    "evaluation/results"
)

RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ---------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------

def save_csv(
    path: Path,
    rows: list[dict],
) -> None:
    """
    Save experiment results to CSV.
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


def clip01(
    value: float,
) -> float:
    """
    Clip a value to the normalized interval [0, 1].
    """

    return max(
        0.0,
        min(
            1.0,
            float(
                value
            ),
        ),
    )


# ---------------------------------------------------------------------
# Imperfect telemetry estimation
# ---------------------------------------------------------------------

def estimate_quality_indicator(
    true_value: float,
    estimation_std: float,
) -> float:
    """
    Generate an imperfect telemetry estimate.

    The physical simulation uses true noise and drift.

    TrustManager receives only noisy estimates of these physical
    conditions.
    """

    estimated_value = (
        true_value
        + random.gauss(
            0.0,
            estimation_std,
        )
    )

    return clip01(
        estimated_value
    )


def estimate_confidence(
    estimated_noise: float,
    estimated_drift: float,
    confidence_error_std: float,
) -> float:
    """
    Generate an imperfect confidence estimate.

    Confidence is correlated with the estimated quality indicators,
    but also contains estimation noise.
    """

    confidence = (
        1.0
        - 0.4
        * estimated_noise
        - 0.4
        * estimated_drift
        + random.gauss(
            0.0,
            confidence_error_std,
        )
    )

    return clip01(
        confidence
    )


# ---------------------------------------------------------------------
# Phase 1:
# Healthy substrate
# ---------------------------------------------------------------------

def get_healthy_state(
    step: int,
) -> tuple[
    float,
    float,
]:
    """
    Simulate a healthy PNN substrate.

    Noise and drift remain low with small natural variations.

    Returns:
        true_noise
        true_drift
    """

    true_noise = (
        0.08
        + random.uniform(
            -0.01,
            0.01,
        )
    )

    true_drift = (
        0.03
        + random.uniform(
            0.0,
            0.03,
        )
    )

    return (
        clip01(
            true_noise
        ),
        clip01(
            true_drift
        ),
    )


# ---------------------------------------------------------------------
# Phase 2:
# Progressive degradation
# ---------------------------------------------------------------------

def get_degradation_state(
    degradation_step: int,
    degradation_duration: int,
) -> tuple[
    float,
    float,
]:
    """
    Simulate progressive physical substrate degradation.

    Drift increases strongly over time.

    Noise also increases gradually.

    The resulting degradation should cause instantaneous trust to
    decrease and eventually trigger the adaptive recalibration logic.
    """

    progress = (
        degradation_step
        / max(
            degradation_duration
            - 1,
            1,
        )
    )

    true_noise = (
        0.08
        + 0.25
        * progress
    )

    true_drift = (
        0.05
        + 0.90
        * progress
    )

    return (
        clip01(
            true_noise
        ),
        clip01(
            true_drift
        ),
    )


# ---------------------------------------------------------------------
# Phase 3:
# Recovery following recalibration
# ---------------------------------------------------------------------

def get_recovery_state(
    recovery_step: int,
    recovery_duration: int,
) -> tuple[
    float,
    float,
]:
    """
    Simulate recovery following a recalibration event.

    Noise and drift progressively decrease toward healthy values.

    Instantaneous trust should recover relatively quickly.

    Historical trust should recover more gradually because the
    exponential moving average retains previous reliability history.
    """

    progress = (
        recovery_step
        / max(
            recovery_duration
            - 1,
            1,
        )
    )

    # Start recovery from a degraded state.
    start_noise = 0.20
    start_drift = 0.40

    # Target state after successful recovery.
    target_noise = 0.08
    target_drift = 0.05

    true_noise = (
        start_noise
        + (
            target_noise
            - start_noise
        )
        * progress
    )

    true_drift = (
        start_drift
        + (
            target_drift
            - start_drift
        )
        * progress
    )

    return (
        clip01(
            true_noise
        ),
        clip01(
            true_drift
        ),
    )


# ---------------------------------------------------------------------
# Calculate one trust observation
# ---------------------------------------------------------------------

def calculate_trust_observation(
    trust_manager: TrustManager,
    true_noise: float,
    true_drift: float,
    delay_s: float,
    telemetry_estimation_std: float,
    confidence_error_std: float,
) -> dict:
    """
    Generate imperfect telemetry and calculate instantaneous trust.

    Ground-truth noise and drift are NOT passed directly to the
    TrustManager.
    """

    # -----------------------------------------------------------------
    # Estimate physical quality indicators
    # -----------------------------------------------------------------

    estimated_noise = (
        estimate_quality_indicator(
            true_value=(
                true_noise
            ),
            estimation_std=(
                telemetry_estimation_std
            ),
        )
    )

    estimated_drift = (
        estimate_quality_indicator(
            true_value=(
                true_drift
            ),
            estimation_std=(
                telemetry_estimation_std
            ),
        )
    )

    # -----------------------------------------------------------------
    # Estimate confidence
    # -----------------------------------------------------------------

    confidence = (
        estimate_confidence(
            estimated_noise=(
                estimated_noise
            ),
            estimated_drift=(
                estimated_drift
            ),
            confidence_error_std=(
                confidence_error_std
            ),
        )
    )

    # -----------------------------------------------------------------
    # Calculate freshness
    # -----------------------------------------------------------------

    freshness = (
        trust_manager.compute_freshness(
            delay_s
        )
    )

    # -----------------------------------------------------------------
    # Calculate instantaneous trust
    # -----------------------------------------------------------------

    current_trust = (
        trust_manager.compute_trust(
            confidence=(
                confidence
            ),
            noise=(
                estimated_noise
            ),
            drift=(
                estimated_drift
            ),
            freshness=(
                freshness
            ),
        )
    )

    return {
        "estimated_noise":
            estimated_noise,

        "estimated_drift":
            estimated_drift,

        "confidence":
            confidence,

        "freshness":
            freshness,

        "current_trust":
            current_trust,
    }


# ---------------------------------------------------------------------
# Main adaptive trust evaluation
# ---------------------------------------------------------------------

def evaluate() -> list[dict]:
    """
    Evaluate the adaptive trust mechanism using a continuous
    three-phase PNN lifecycle.

    Phase 1:
        Healthy operation.

    Phase 2:
        Progressive degradation.

    Phase 3:
        Recovery following an actual recalibration event.

    Recovery is executed ONLY when the adaptive mechanism has
    requested recalibration.

    Recalibration can be triggered by either:

        - persistent low historical trust, or
        - several consecutive low instantaneous-trust observations.
    """

    # -----------------------------------------------------------------
    # Reproducibility
    # -----------------------------------------------------------------

    random.seed(
        42
    )

    # -----------------------------------------------------------------
    # Edge-level trust configuration
    # -----------------------------------------------------------------

    trust_manager = (
        TrustManager(
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
    )

    # -----------------------------------------------------------------
    # Adaptive historical trust configuration
    # -----------------------------------------------------------------

    adaptive_tracker = (
        AdaptiveTrustTracker(
            AdaptiveTrustConfig(

                # EMA memory factor.
                lambda_history=0.8,

                # Current trust below this value counts as a
                # low-trust observation.
                low_trust_threshold=0.60,

                # Historical trust below this value indicates
                # persistent long-term degradation.
                historical_trust_threshold=0.60,

                # Three consecutive low-trust observations may
                # independently trigger recalibration.
                low_trust_streak_limit=3,
            )
        )
    )

    # -----------------------------------------------------------------
    # Experimental configuration
    # -----------------------------------------------------------------

    backend_id = (
        "adaptive-pnn-backend"
    )

    healthy_duration = 30

    # Maximum length of degradation.
    #
    # The phase ends earlier when recalibration is triggered.
    degradation_duration = 60

    recovery_duration = 30

    telemetry_estimation_std = (
        0.05
    )

    confidence_error_std = (
        0.08
    )

    # Keep freshness approximately constant so the experiment
    # primarily measures degradation and adaptive reliability.
    delay_s = (
        0.5
    )

    rows: list[
        dict
    ] = []

    global_step = (
        0
    )

    recalibration_step: (
        int
        | None
    ) = None

    recalibration_reason: (
        str
        | None
    ) = None

    # =================================================================
    # PHASE 1:
    # HEALTHY OPERATION
    # =================================================================

    for phase_step in range(
        healthy_duration
    ):

        # -------------------------------------------------------------
        # Generate hidden physical state
        # -------------------------------------------------------------

        (
            true_noise,
            true_drift,
        ) = (
            get_healthy_state(
                phase_step
            )
        )

        # -------------------------------------------------------------
        # Generate imperfect telemetry and current trust
        # -------------------------------------------------------------

        observation = (
            calculate_trust_observation(
                trust_manager=(
                    trust_manager
                ),
                true_noise=(
                    true_noise
                ),
                true_drift=(
                    true_drift
                ),
                delay_s=(
                    delay_s
                ),
                telemetry_estimation_std=(
                    telemetry_estimation_std
                ),
                confidence_error_std=(
                    confidence_error_std
                ),
            )
        )

        current_trust = (
            observation[
                "current_trust"
            ]
        )

        # -------------------------------------------------------------
        # Update historical trust
        # -------------------------------------------------------------

        historical_trust = (
            adaptive_tracker.update(
                backend_id=(
                    backend_id
                ),
                current_trust=(
                    current_trust
                ),
            )
        )

        low_trust_streak = (
            adaptive_tracker
            .get_low_trust_streak(
                backend_id
            )
        )

        # Healthy phase should not normally trigger recalibration,
        # but we record the mechanism state for completeness.
        recalibration_requested = (
            adaptive_tracker
            .requires_recalibration(
                backend_id
            )
        )

        # -------------------------------------------------------------
        # Store result
        # -------------------------------------------------------------

        rows.append(
            {
                "step":
                    global_step,

                "phase":
                    "healthy",

                "phase_step":
                    phase_step,

                "true_noise":
                    true_noise,

                "estimated_noise":
                    observation[
                        "estimated_noise"
                    ],

                "true_drift":
                    true_drift,

                "estimated_drift":
                    observation[
                        "estimated_drift"
                    ],

                "confidence":
                    observation[
                        "confidence"
                    ],

                "freshness":
                    observation[
                        "freshness"
                    ],

                "current_trust":
                    current_trust,

                "historical_trust":
                    historical_trust,

                "low_trust_streak":
                    low_trust_streak,

                "recalibration_requested":
                    recalibration_requested,

                "recalibration_event":
                    False,

                "recalibration_reason":
                    "",
            }
        )

        global_step += (
            1
        )

    # =================================================================
    # PHASE 2:
    # PROGRESSIVE DEGRADATION
    # =================================================================

    degradation_steps_executed = (
        0
    )

    for phase_step in range(
        degradation_duration
    ):

        degradation_steps_executed += (
            1
        )

        # -------------------------------------------------------------
        # Generate progressively degraded physical state
        # -------------------------------------------------------------

        (
            true_noise,
            true_drift,
        ) = (
            get_degradation_state(
                degradation_step=(
                    phase_step
                ),
                degradation_duration=(
                    degradation_duration
                ),
            )
        )

        # -------------------------------------------------------------
        # Generate imperfect telemetry and instantaneous trust
        # -------------------------------------------------------------

        observation = (
            calculate_trust_observation(
                trust_manager=(
                    trust_manager
                ),
                true_noise=(
                    true_noise
                ),
                true_drift=(
                    true_drift
                ),
                delay_s=(
                    delay_s
                ),
                telemetry_estimation_std=(
                    telemetry_estimation_std
                ),
                confidence_error_std=(
                    confidence_error_std
                ),
            )
        )

        current_trust = (
            observation[
                "current_trust"
            ]
        )

        # -------------------------------------------------------------
        # Update historical trust
        # -------------------------------------------------------------

        historical_trust = (
            adaptive_tracker.update(
                backend_id=(
                    backend_id
                ),
                current_trust=(
                    current_trust
                ),
            )
        )

        low_trust_streak = (
            adaptive_tracker
            .get_low_trust_streak(
                backend_id
            )
        )

        # -------------------------------------------------------------
        # Check the two recalibration conditions separately.
        #
        # This lets us record why recalibration occurred.
        # -------------------------------------------------------------

        historical_degradation = (
            historical_trust
            < adaptive_tracker
            .config
            .historical_trust_threshold
        )

        repeated_low_trust = (
            low_trust_streak
            >= adaptive_tracker
            .config
            .low_trust_streak_limit
        )

        recalibration_requested = (
            adaptive_tracker
            .requires_recalibration(
                backend_id
            )
        )

        # -------------------------------------------------------------
        # Only the first recalibration request becomes an event.
        # -------------------------------------------------------------

        recalibration_event = (
            recalibration_requested
            and recalibration_step
            is None
        )

        current_recalibration_reason = (
            ""
        )

        if recalibration_event:

            recalibration_step = (
                global_step
            )

            if (
                historical_degradation
                and repeated_low_trust
            ):

                recalibration_reason = (
                    "historical_trust_and_low_trust_streak"
                )

            elif (
                historical_degradation
            ):

                recalibration_reason = (
                    "historical_trust"
                )

            elif (
                repeated_low_trust
            ):

                recalibration_reason = (
                    "low_trust_streak"
                )

            else:

                recalibration_reason = (
                    "adaptive_trust_condition"
                )

            current_recalibration_reason = (
                recalibration_reason
            )

        # -------------------------------------------------------------
        # Store degradation observation
        # -------------------------------------------------------------

        rows.append(
            {
                "step":
                    global_step,

                "phase":
                    "degradation",

                "phase_step":
                    phase_step,

                "true_noise":
                    true_noise,

                "estimated_noise":
                    observation[
                        "estimated_noise"
                    ],

                "true_drift":
                    true_drift,

                "estimated_drift":
                    observation[
                        "estimated_drift"
                    ],

                "confidence":
                    observation[
                        "confidence"
                    ],

                "freshness":
                    observation[
                        "freshness"
                    ],

                "current_trust":
                    current_trust,

                "historical_trust":
                    historical_trust,

                "low_trust_streak":
                    low_trust_streak,

                "recalibration_requested":
                    recalibration_requested,

                "recalibration_event":
                    recalibration_event,

                "recalibration_reason":
                    current_recalibration_reason,
            }
        )

        global_step += (
            1
        )

        # -------------------------------------------------------------
        # Stop degradation immediately after actual recalibration.
        # -------------------------------------------------------------

        if recalibration_event:
            break

    # =================================================================
    # PHASE 3:
    # RECOVERY FOLLOWING RECALIBRATION
    #
    # IMPORTANT:
    # This phase executes ONLY when recalibration was really triggered.
    # =================================================================

    if (
        recalibration_step
        is not None
    ):

        # Reset the short-term low-trust streak after recalibration.
        #
        # Historical trust is preserved so recovery must be earned
        # gradually through subsequent reliable observations.
        adaptive_tracker.reset_after_recalibration(
            backend_id
        )

        for phase_step in range(
            recovery_duration
        ):

            # ---------------------------------------------------------
            # Generate recovering physical state
            # ---------------------------------------------------------

            (
                true_noise,
                true_drift,
            ) = (
                get_recovery_state(
                    recovery_step=(
                        phase_step
                    ),
                    recovery_duration=(
                        recovery_duration
                    ),
                )
            )

            # ---------------------------------------------------------
            # Generate imperfect telemetry and trust
            # ---------------------------------------------------------

            observation = (
                calculate_trust_observation(
                    trust_manager=(
                        trust_manager
                    ),
                    true_noise=(
                        true_noise
                    ),
                    true_drift=(
                        true_drift
                    ),
                    delay_s=(
                        delay_s
                    ),
                    telemetry_estimation_std=(
                        telemetry_estimation_std
                    ),
                    confidence_error_std=(
                        confidence_error_std
                    ),
                )
            )

            current_trust = (
                observation[
                    "current_trust"
                ]
            )

            # ---------------------------------------------------------
            # Historical trust recovers gradually through EMA
            # ---------------------------------------------------------

            historical_trust = (
                adaptive_tracker.update(
                    backend_id=(
                        backend_id
                    ),
                    current_trust=(
                        current_trust
                    ),
                )
            )

            low_trust_streak = (
                adaptive_tracker
                .get_low_trust_streak(
                    backend_id
                )
            )

            # ---------------------------------------------------------
            # During recovery we record whether the reliability
            # mechanism still considers recalibration necessary.
            #
            # We do NOT trigger a second recalibration event in this
            # experiment because the objective is to observe recovery
            # after the first recalibration.
            # ---------------------------------------------------------

            recalibration_requested = (
                adaptive_tracker
                .requires_recalibration(
                    backend_id
                )
            )

            # ---------------------------------------------------------
            # Store recovery observation
            # ---------------------------------------------------------

            rows.append(
                {
                    "step":
                        global_step,

                    "phase":
                        "recovery",

                    "phase_step":
                        phase_step,

                    "true_noise":
                        true_noise,

                    "estimated_noise":
                        observation[
                            "estimated_noise"
                        ],

                    "true_drift":
                        true_drift,

                    "estimated_drift":
                        observation[
                            "estimated_drift"
                        ],

                    "confidence":
                        observation[
                            "confidence"
                        ],

                    "freshness":
                        observation[
                            "freshness"
                        ],

                    "current_trust":
                        current_trust,

                    "historical_trust":
                        historical_trust,

                    "low_trust_streak":
                        low_trust_streak,

                    "recalibration_requested":
                        recalibration_requested,

                    "recalibration_event":
                        False,

                    "recalibration_reason":
                        "",
                }
            )

            global_step += (
                1
            )

    else:

        print()

        print(
            "WARNING: No recalibration was triggered "
            "during degradation."
        )

        print(
            "The recovery phase was therefore NOT executed."
        )

    # -----------------------------------------------------------------
    # Console summary
    # -----------------------------------------------------------------

    print()

    print(
        "Adaptive trust evaluation completed."
    )

    if (
        recalibration_step
        is not None
    ):

        print(
            "Recalibration triggered at "
            f"global step: "
            f"{recalibration_step}"
        )

        print(
            "Recalibration reason: "
            f"{recalibration_reason}"
        )

    print(
        "Degradation steps executed: "
        f"{degradation_steps_executed}"
    )

    return (
        rows
    )


# ---------------------------------------------------------------------
# Phase-level summary
# ---------------------------------------------------------------------

def summarize_phases(
    rows: list[
        dict
    ],
) -> list[
    dict
]:
    """
    Calculate summary statistics for healthy, degradation,
    and recovery phases.
    """

    phases = [
        "healthy",
        "degradation",
        "recovery",
    ]

    summary_rows: list[
        dict
    ] = []

    for phase in phases:

        phase_rows = [
            row
            for row
            in rows
            if row[
                "phase"
            ]
            == phase
        ]

        # Recovery may legitimately not exist if recalibration
        # was never triggered.
        if not phase_rows:
            continue

        number_of_steps = (
            len(
                phase_rows
            )
        )

        mean_current_trust = (
            sum(
                float(
                    row[
                        "current_trust"
                    ]
                )
                for row
                in phase_rows
            )
            / number_of_steps
        )

        mean_historical_trust = (
            sum(
                float(
                    row[
                        "historical_trust"
                    ]
                )
                for row
                in phase_rows
            )
            / number_of_steps
        )

        minimum_current_trust = (
            min(
                float(
                    row[
                        "current_trust"
                    ]
                )
                for row
                in phase_rows
            )
        )

        maximum_current_trust = (
            max(
                float(
                    row[
                        "current_trust"
                    ]
                )
                for row
                in phase_rows
            )
        )

        minimum_historical_trust = (
            min(
                float(
                    row[
                        "historical_trust"
                    ]
                )
                for row
                in phase_rows
            )
        )

        maximum_historical_trust = (
            max(
                float(
                    row[
                        "historical_trust"
                    ]
                )
                for row
                in phase_rows
            )
        )

        mean_true_noise = (
            sum(
                float(
                    row[
                        "true_noise"
                    ]
                )
                for row
                in phase_rows
            )
            / number_of_steps
        )

        mean_estimated_noise = (
            sum(
                float(
                    row[
                        "estimated_noise"
                    ]
                )
                for row
                in phase_rows
            )
            / number_of_steps
        )

        mean_true_drift = (
            sum(
                float(
                    row[
                        "true_drift"
                    ]
                )
                for row
                in phase_rows
            )
            / number_of_steps
        )

        mean_estimated_drift = (
            sum(
                float(
                    row[
                        "estimated_drift"
                    ]
                )
                for row
                in phase_rows
            )
            / number_of_steps
        )

        recalibration_events = (
            sum(
                1
                for row
                in phase_rows
                if bool(
                    row[
                        "recalibration_event"
                    ]
                )
            )
        )

        maximum_low_trust_streak = (
            max(
                int(
                    row[
                        "low_trust_streak"
                    ]
                )
                for row
                in phase_rows
            )
        )

        summary_rows.append(
            {
                "phase":
                    phase,

                "steps":
                    number_of_steps,

                "mean_current_trust":
                    mean_current_trust,

                "min_current_trust":
                    minimum_current_trust,

                "max_current_trust":
                    maximum_current_trust,

                "mean_historical_trust":
                    mean_historical_trust,

                "min_historical_trust":
                    minimum_historical_trust,

                "max_historical_trust":
                    maximum_historical_trust,

                "mean_true_noise":
                    mean_true_noise,

                "mean_estimated_noise":
                    mean_estimated_noise,

                "mean_true_drift":
                    mean_true_drift,

                "mean_estimated_drift":
                    mean_estimated_drift,

                "max_low_trust_streak":
                    maximum_low_trust_streak,

                "recalibration_events":
                    recalibration_events,
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
    Run adaptive trust evaluation and save:

        adaptive_trust_details.csv
        adaptive_trust_summary.csv
    """

    rows = (
        evaluate()
    )

    summary_rows = (
        summarize_phases(
            rows
        )
    )

    details_path = (
        RESULTS_DIR
        / "adaptive_trust_details.csv"
    )

    summary_path = (
        RESULTS_DIR
        / "adaptive_trust_summary.csv"
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
        "Saved adaptive trust details to: "
        f"{details_path}"
    )

    print(
        "Saved adaptive trust summary to: "
        f"{summary_path}"
    )


if __name__ == "__main__":
    main()