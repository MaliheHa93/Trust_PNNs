from __future__ import annotations

import csv
import random
from pathlib import Path
from statistics import median

from core.fog_fusion import (
    FogFusionConfig,
    FogFusionEngine,
)
from core.trust_models import (
    EvidenceRecord,
)
from core.trust_manager import (
    TrustConfig,
    TrustManager,
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
# Utility functions
# ---------------------------------------------------------------------

def save_csv(
    path: Path,
    rows: list[dict],
) -> None:
    """
    Save experimental results to a CSV file.
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
    Clip a numeric value to [0, 1].
    """

    return max(
        0.0,
        min(
            1.0,
            float(value),
        ),
    )


# ---------------------------------------------------------------------
# Baseline 1:
# Simple averaging
# ---------------------------------------------------------------------

def simple_average(
    values: list[float],
) -> float:
    """
    Compute the unweighted average.

        w_i = 1
    """

    if not values:
        raise ValueError(
            "Cannot average an empty list."
        )

    return (
        sum(
            values
        )
        / len(
            values
        )
    )


# ---------------------------------------------------------------------
# Baseline 2:
# Confidence-weighted averaging
# ---------------------------------------------------------------------

def confidence_weighted_average(
    values: list[float],
    confidences: list[float],
) -> float:
    """
    Fuse outputs according to confidence.

        w_i = c_i
    """

    if (
        len(
            values
        )
        != len(
            confidences
        )
    ):
        raise ValueError(
            "Values and confidences "
            "must have the same length."
        )

    if not values:
        raise ValueError(
            "Cannot fuse an empty list."
        )

    denominator = sum(
        confidences
    )

    if denominator <= 0.0:

        return simple_average(
            values
        )

    numerator = sum(

        value
        * confidence

        for (
            value,
            confidence,
        )

        in zip(
            values,
            confidences,
        )
    )

    return (
        numerator
        / denominator
    )


# ---------------------------------------------------------------------
# Baseline 3:
# Uncertainty-weighted averaging
# ---------------------------------------------------------------------

def uncertainty_weighted_average(
    values: list[float],
    noises: list[float],
    sigma_floor: float = 0.02,
    epsilon: float = 1e-6,
) -> float:
    """
    Fuse outputs using inverse estimated uncertainty.

        w_i =
            1 /
            (max(sigma_i, sigma_floor)^2 + epsilon)

    sigma_floor prevents near-zero estimated uncertainty from
    producing unrealistically extreme weights.

    The same floor is used by FogFusionEngine so that the baseline
    and proposed methods are compared fairly.
    """

    if (
        len(
            values
        )
        != len(
            noises
        )
    ):
        raise ValueError(
            "Values and noises "
            "must have the same length."
        )

    if not values:
        raise ValueError(
            "Cannot fuse an empty list."
        )

    weights: list[
        float
    ] = []

    for noise in noises:

        effective_noise = max(
            float(
                noise
            ),
            sigma_floor,
        )

        weight = (
            1.0
            / (
                effective_noise ** 2
                + epsilon
            )
        )

        weights.append(
            weight
        )

    denominator = sum(
        weights
    )

    if denominator <= 0.0:

        return simple_average(
            values
        )

    numerator = sum(

        value
        * weight

        for (
            value,
            weight,
        )

        in zip(
            values,
            weights,
        )
    )

    return (
        numerator
        / denominator
    )


# ---------------------------------------------------------------------
# Baseline 4:
# Trust-only weighted averaging
# ---------------------------------------------------------------------

def trust_weighted_average(
    values: list[float],
    trusts: list[float],
) -> float:
    """
    Fuse outputs using only edge-level trust.

        w_i = T_i
    """

    if (
        len(
            values
        )
        != len(
            trusts
        )
    ):
        raise ValueError(
            "Values and trust scores "
            "must have the same length."
        )

    if not values:
        raise ValueError(
            "Cannot fuse an empty list."
        )

    denominator = sum(
        trusts
    )

    if denominator <= 0.0:

        return simple_average(
            values
        )

    numerator = sum(

        value
        * trust

        for (
            value,
            trust,
        )

        in zip(
            values,
            trusts,
        )
    )

    return (
        numerator
        / denominator
    )


# ---------------------------------------------------------------------
# Baseline 5:
# Median fusion
# ---------------------------------------------------------------------

def median_fusion(
    values: list[float],
) -> float:
    """
    Robust non-parametric fusion baseline.

    Median fusion is less sensitive to extreme numerical outliers.
    """

    if not values:
        raise ValueError(
            "Cannot fuse an empty list."
        )

    return float(
        median(
            values
        )
    )


# ---------------------------------------------------------------------
# Imperfect telemetry estimation
# ---------------------------------------------------------------------

def estimate_quality_indicator(
    true_value: float,
    estimation_std: float,
) -> float:
    """
    Generate an imperfect estimate of a hidden physical condition.

    Ground-truth noise and drift are used to generate the physical
    output.

    TrustManager receives only these noisy estimates.
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


# ---------------------------------------------------------------------
# Improved confidence estimation
# ---------------------------------------------------------------------

def estimate_confidence(
    backend_quality: float,
    estimated_noise: float,
    estimated_drift: float,
    confidence_error_std: float,
) -> float:
    """
    Generate confidence from multiple factors.

    Confidence is primarily based on an independent backend-quality
    variable.

    Noise and drift have smaller contributions.

    This avoids making confidence simply another representation of
    noise and drift.
    """

    confidence = (

        0.60
        * backend_quality

        + 0.20
        * (
            1.0
            - estimated_noise
        )

        + 0.20
        * (
            1.0
            - estimated_drift
        )

        + random.gauss(
            0.0,
            confidence_error_std,
        )
    )

    return clip01(
        confidence
    )


# ---------------------------------------------------------------------
# Signed physical drift
# ---------------------------------------------------------------------

def generate_signed_drift(
    drift_magnitude: float,
) -> float:
    """
    Generate physical drift in either direction.

    The trust layer receives drift magnitude, while the physical
    output may drift positively or negatively.
    """

    direction = random.choice(
        [
            -1.0,
            1.0,
        ]
    )

    return (
        clip01(
            drift_magnitude
        )
        * direction
    )


# ---------------------------------------------------------------------
# Fog-fusion evaluation
# ---------------------------------------------------------------------

def evaluate(
    seed: int = 42,
) -> list[dict]:
    """
    Run the fog-level fusion experiment for one random seed.

    The function returns all scenario-level results without writing
    a CSV file.

    This allows the same experiment to be reused by the multi-seed
    statistical evaluation.

    Normal single-seed execution uses seed=42.
    """

    # -----------------------------------------------------------------
    # Reproducibility
    # -----------------------------------------------------------------

    random.seed(
        seed
    )

    # -----------------------------------------------------------------
    # Ground-truth state at decision time
    # -----------------------------------------------------------------

    true_value = (
        1.0
    )

    # -----------------------------------------------------------------
    # Number of participating PNN sources
    # -----------------------------------------------------------------

    source_counts = [
        2,
        3,
        5,
        7,
    ]

    # -----------------------------------------------------------------
    # Controlled base physical-noise levels
    # -----------------------------------------------------------------

    noise_levels = [
        0.05,
        0.10,
        0.20,
        0.30,
        0.40,
    ]

    # -----------------------------------------------------------------
    # Controlled base physical-drift magnitudes
    # -----------------------------------------------------------------

    drift_levels = [
        0.0,
        0.10,
        0.20,
        0.30,
    ]

    # -----------------------------------------------------------------
    # Repetitions per condition
    # -----------------------------------------------------------------

    repetitions = (
        50
    )

    # -----------------------------------------------------------------
    # Imperfect telemetry parameters
    # -----------------------------------------------------------------

    telemetry_estimation_std = (
        0.05
    )

    confidence_error_std = (
        0.05
    )

    # -----------------------------------------------------------------
    # Temporal state evolution
    #
    # The current ground-truth state is true_value.
    #
    # A PNN output generated delay_s seconds earlier may represent
    # a slightly different physical state.
    #
    # This makes freshness relevant to actual output quality.
    # -----------------------------------------------------------------

    max_state_change_rate_per_s = (
        0.02
    )

    # -----------------------------------------------------------------
    # Trust configuration
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
    # Fog fusion configuration
    # -----------------------------------------------------------------

    fusion_engine = (
        FogFusionEngine(

            FogFusionConfig(

                # Avoid unrealistically large weights when an
                # uncertainty estimate is clipped near zero.

                sigma_floor=0.02,

                epsilon=1e-6,

                # Source delays range from 0.1 to 5 seconds.
                # Therefore all sources from the same simulated
                # observation cycle remain temporally compatible.

                max_timestamp_difference_s=5.0,

                # Evidence exceeding this disagreement threshold is
                # not fused operationally.

                max_normalized_disagreement=0.75,

                # Reliability factor R_i excludes noise because noise
                # is already used by inverse-uncertainty weighting.

                confidence_weight=
                    1.0 / 3.0,

                drift_weight=
                    1.0 / 3.0,

                freshness_weight=
                    1.0 / 3.0,
            )
        )
    )

    rows: list[
        dict
    ] = []

    # =================================================================
    # Experimental loops
    # =================================================================

    for source_count in (
        source_counts
    ):

        for noise_level in (
            noise_levels
        ):

            for drift_level in (
                drift_levels
            ):

                for repetition in range(
                    repetitions
                ):

                    # -------------------------------------------------
                    # Every repetition represents one common decision
                    # context / observation window.
                    #
                    # Include seed in the identifier so the same
                    # configuration from different seeds remains
                    # uniquely traceable.
                    # -------------------------------------------------

                    observation_window_id = (
                        f"seed-{seed}"
                        f"-sources-{source_count}"
                        f"-noise-{noise_level}"
                        f"-drift-{drift_level}"
                        f"-rep-{repetition}"
                    )

                    decision_context = (
                        "fog-fusion-evaluation"
                    )

                    # -------------------------------------------------
                    # Common reference timestamp.
                    # -------------------------------------------------

                    reference_timestamp = (
                        1_000_000.0
                        + repetition
                    )

                    # -------------------------------------------------
                    # Randomize source-quality ordering.
                    # -------------------------------------------------

                    quality_ranks = list(
                        range(
                            source_count
                        )
                    )

                    random.shuffle(
                        quality_ranks
                    )

                    # -------------------------------------------------
                    # Containers
                    # -------------------------------------------------

                    records: list[
                        EvidenceRecord
                    ] = []

                    values: list[
                        float
                    ] = []

                    confidences: list[
                        float
                    ] = []

                    estimated_noises: list[
                        float
                    ] = []

                    estimated_drifts: list[
                        float
                    ] = []

                    trusts: list[
                        float
                    ] = []

                    true_noises: list[
                        float
                    ] = []

                    true_drifts: list[
                        float
                    ] = []

                    signed_drifts: list[
                        float
                    ] = []

                    backend_qualities: list[
                        float
                    ] = []

                    staleness_errors: list[
                        float
                    ] = []

                    # =================================================
                    # Generate individual PNN sources
                    # =================================================

                    for source_index in range(
                        source_count
                    ):

                        # ---------------------------------------------
                        # Randomized heterogeneity rank
                        # ---------------------------------------------

                        quality_rank = (
                            quality_ranks[
                                source_index
                            ]
                        )

                        normalized_rank = (
                            quality_rank
                            / max(
                                source_count
                                - 1,
                                1,
                            )
                        )

                        # =============================================
                        # STEP 1:
                        # Hidden physical noise
                        # =============================================

                        true_source_noise = (

                            noise_level

                            * (
                                1.0
                                + 0.25
                                * quality_rank
                            )
                        )

                        true_source_noise = (
                            clip01(
                                true_source_noise
                            )
                        )

                        # =============================================
                        # STEP 2:
                        # Hidden physical drift magnitude
                        # =============================================

                        true_source_drift = (

                            drift_level

                            * normalized_rank
                        )

                        true_source_drift = (
                            clip01(
                                true_source_drift
                            )
                        )

                        # ---------------------------------------------
                        # Drift may occur in either direction.
                        # ---------------------------------------------

                        signed_source_drift = (
                            generate_signed_drift(
                                true_source_drift
                            )
                        )

                        # =============================================
                        # STEP 3:
                        # Source delay
                        # =============================================

                        delay_s = (
                            random.choice(
                                [
                                    0.1,
                                    1.0,
                                    3.0,
                                    5.0,
                                ]
                            )
                        )

                        # =============================================
                        # STEP 4:
                        # Temporal evolution of the physical state
                        # =============================================

                        state_velocity = (
                            random.uniform(

                                -max_state_change_rate_per_s,

                                max_state_change_rate_per_s,
                            )
                        )

                        value_at_generation = (

                            true_value

                            - state_velocity
                            * delay_s
                        )

                        staleness_error = abs(

                            true_value

                            - value_at_generation
                        )

                        # =============================================
                        # STEP 5:
                        # Generate physical output
                        # =============================================

                        random_noise = (
                            random.gauss(
                                0.0,
                                true_source_noise,
                            )
                        )

                        value = (

                            value_at_generation

                            + random_noise

                            + signed_source_drift
                        )

                        # =============================================
                        # STEP 6:
                        # Imperfect noise telemetry
                        # =============================================

                        estimated_noise = (
                            estimate_quality_indicator(

                                true_value=
                                    true_source_noise,

                                estimation_std=
                                    telemetry_estimation_std,
                            )
                        )

                        # =============================================
                        # STEP 7:
                        # Imperfect drift telemetry
                        # =============================================

                        estimated_drift = (
                            estimate_quality_indicator(

                                true_value=
                                    true_source_drift,

                                estimation_std=
                                    telemetry_estimation_std,
                            )
                        )

                        # =============================================
                        # STEP 8:
                        # Independent latent substrate quality
                        # =============================================

                        backend_quality = (
                            random.uniform(
                                0.30,
                                1.0,
                            )
                        )

                        # =============================================
                        # STEP 9:
                        # Generate confidence
                        # =============================================

                        confidence = (
                            estimate_confidence(

                                backend_quality=
                                    backend_quality,

                                estimated_noise=
                                    estimated_noise,

                                estimated_drift=
                                    estimated_drift,

                                confidence_error_std=
                                    confidence_error_std,
                            )
                        )

                        # =============================================
                        # STEP 10:
                        # Freshness
                        # =============================================

                        freshness = (
                            trust_manager
                            .compute_freshness(
                                delay_s
                            )
                        )

                        # =============================================
                        # STEP 11:
                        # Edge-level trust
                        # =============================================

                        trust = (
                            trust_manager
                            .compute_trust(

                                confidence=
                                    confidence,

                                noise=
                                    estimated_noise,

                                drift=
                                    estimated_drift,

                                freshness=
                                    freshness,
                            )
                        )

                        # =============================================
                        # STEP 12:
                        # Evidence timestamp
                        # =============================================

                        timestamp = (

                            reference_timestamp

                            - delay_s
                        )

                        # =============================================
                        # STEP 13:
                        # Build evidence record
                        # =============================================

                        record = (
                            EvidenceRecord(

                                backend_id=(
                                    f"synthetic-pnn-"
                                    f"{source_index}"
                                ),

                                task_id=(
                                    "fusion-task"
                                ),

                                value=
                                    value,

                                confidence=
                                    confidence,

                                noise=
                                    estimated_noise,

                                drift=
                                    estimated_drift,

                                timestamp=
                                    timestamp,

                                modality=(
                                    "synthetic_numeric"
                                ),

                                provenance={

                                    "seed":
                                        seed,

                                    "source_index":
                                        source_index,

                                    "quality_rank":
                                        quality_rank,

                                    "true_noise":
                                        true_source_noise,

                                    "true_drift":
                                        true_source_drift,

                                    "signed_drift":
                                        signed_source_drift,

                                    "backend_quality":
                                        backend_quality,

                                    "state_velocity":
                                        state_velocity,

                                    "staleness_error":
                                        staleness_error,

                                    "decision_context":
                                        decision_context,

                                    "observation_window_id":
                                        observation_window_id,
                                },

                                freshness=
                                    freshness,

                                trust=
                                    trust,
                            )
                        )

                        # =============================================
                        # Store source information
                        # =============================================

                        records.append(
                            record
                        )

                        values.append(
                            value
                        )

                        confidences.append(
                            confidence
                        )

                        estimated_noises.append(
                            estimated_noise
                        )

                        estimated_drifts.append(
                            estimated_drift
                        )

                        trusts.append(
                            trust
                        )

                        true_noises.append(
                            true_source_noise
                        )

                        true_drifts.append(
                            true_source_drift
                        )

                        signed_drifts.append(
                            signed_source_drift
                        )

                        backend_qualities.append(
                            backend_quality
                        )

                        staleness_errors.append(
                            staleness_error
                        )

                    # =================================================
                    # Fog compatibility and disagreement assessment
                    # =================================================

                    assessment = (
                        fusion_engine
                        .assess(
                            records
                        )
                    )

                    # =================================================
                    # METHOD 1:
                    # Simple average
                    # =================================================

                    simple_result = (
                        simple_average(
                            values
                        )
                    )

                    # =================================================
                    # METHOD 2:
                    # Confidence weighted
                    # =================================================

                    confidence_result = (
                        confidence_weighted_average(

                            values=
                                values,

                            confidences=
                                confidences,
                        )
                    )

                    # =================================================
                    # METHOD 3:
                    # Uncertainty weighted
                    # =================================================

                    uncertainty_result = (
                        uncertainty_weighted_average(

                            values=
                                values,

                            noises=
                                estimated_noises,

                            sigma_floor=(
                                fusion_engine
                                .config
                                .sigma_floor
                            ),

                            epsilon=(
                                fusion_engine
                                .config
                                .epsilon
                            ),
                        )
                    )

                    # =================================================
                    # METHOD 4:
                    # Trust only
                    # =================================================

                    trust_only_result = (
                        trust_weighted_average(

                            values=
                                values,

                            trusts=
                                trusts,
                        )
                    )

                    # =================================================
                    # METHOD 5:
                    # Median
                    # =================================================

                    median_result = (
                        median_fusion(
                            values
                        )
                    )

                    # =================================================
                    # METHOD 6:
                    # Original T / sigma^2
                    # =================================================

                    original_trust_uncertainty_result = (
                        fusion_engine
                        .fuse_numeric_original_trust(

                            records,

                            enforce_assessment=
                                False,
                        )
                    )

                    # =================================================
                    # METHOD 7:
                    # Proposed R / sigma^2
                    # =================================================

                    orthogonal_trust_uncertainty_result = (
                        fusion_engine
                        .fuse_numeric(

                            records,

                            enforce_assessment=
                                False,
                        )
                    )

                    # =================================================
                    # Operational fog behavior
                    # =================================================

                    if not (
                        assessment.compatible
                    ):

                        fog_action = (
                            "reject_incompatible_evidence"
                        )

                        operational_fusion_result = (
                            None
                        )

                    elif not (
                        assessment
                        .disagreement_acceptable
                    ):

                        fog_action = (
                            "request_additional_measurement_or_recalibration"
                        )

                        operational_fusion_result = (
                            None
                        )

                    else:

                        fog_action = (
                            "fuse"
                        )

                        operational_fusion_result = (
                            orthogonal_trust_uncertainty_result
                        )

                    # =================================================
                    # Calculate numerical errors
                    # =================================================

                    simple_error = abs(
                        simple_result
                        - true_value
                    )

                    confidence_error = abs(
                        confidence_result
                        - true_value
                    )

                    uncertainty_error = abs(
                        uncertainty_result
                        - true_value
                    )

                    trust_only_error = abs(
                        trust_only_result
                        - true_value
                    )

                    median_error = abs(
                        median_result
                        - true_value
                    )

                    original_trust_uncertainty_error = abs(

                        original_trust_uncertainty_result

                        - true_value
                    )

                    orthogonal_trust_uncertainty_error = abs(

                        orthogonal_trust_uncertainty_result

                        - true_value
                    )

                    if (
                        operational_fusion_result
                        is not None
                    ):

                        operational_fusion_error = abs(

                            operational_fusion_result

                            - true_value
                        )

                    else:

                        operational_fusion_error = (
                            None
                        )

                    # =================================================
                    # Save result
                    # =================================================

                    rows.append(
                        {
                            # -----------------------------------------
                            # Random seed
                            # -----------------------------------------

                            "seed":
                                seed,

                            # -----------------------------------------
                            # Experimental configuration
                            # -----------------------------------------

                            "source_count":
                                source_count,

                            "noise_level":
                                noise_level,

                            "drift_level":
                                drift_level,

                            "repetition":
                                repetition,

                            # =========================================
                            # Method 1:
                            # Simple average
                            # =========================================

                            "simple_average_error":
                                simple_error,

                            "simple_average_result":
                                simple_result,

                            # =========================================
                            # Method 2:
                            # Confidence weighted
                            # =========================================

                            "confidence_weighted_error":
                                confidence_error,

                            "confidence_weighted_result":
                                confidence_result,

                            # =========================================
                            # Method 3:
                            # Uncertainty weighted
                            # =========================================

                            "uncertainty_weighted_error":
                                uncertainty_error,

                            "uncertainty_weighted_result":
                                uncertainty_result,

                            # =========================================
                            # Method 4:
                            # Trust only
                            # =========================================

                            "trust_only_error":
                                trust_only_error,

                            "trust_only_result":
                                trust_only_result,

                            # =========================================
                            # Method 5:
                            # Median
                            # =========================================

                            "median_error":
                                median_error,

                            "median_result":
                                median_result,

                            # =========================================
                            # Method 6:
                            # Original T / sigma^2
                            # =========================================

                            "original_trust_uncertainty_error":
                                original_trust_uncertainty_error,

                            "original_trust_uncertainty_result":
                                original_trust_uncertainty_result,

                            # =========================================
                            # Method 7:
                            # Proposed R / sigma^2
                            # =========================================

                            "orthogonal_trust_uncertainty_error":
                                orthogonal_trust_uncertainty_error,

                            "orthogonal_trust_uncertainty_result":
                                orthogonal_trust_uncertainty_result,

                            # -----------------------------------------
                            # Backward-compatible aliases
                            # -----------------------------------------

                            "trust_aware_error":
                                orthogonal_trust_uncertainty_error,

                            "trust_aware_result":
                                orthogonal_trust_uncertainty_result,

                            # =========================================
                            # Operational fog decision
                            # =========================================

                            "compatible":
                                assessment.compatible,

                            "compatibility_issue_count":
                                len(
                                    assessment
                                    .compatibility_issues
                                ),

                            "compatibility_issues":
                                " | ".join(
                                    assessment
                                    .compatibility_issues
                                ),

                            "disagreement":
                                assessment.disagreement,

                            "disagreement_acceptable":
                                assessment
                                .disagreement_acceptable,

                            "can_fuse":
                                assessment.can_fuse,

                            "fog_action":
                                fog_action,

                            "operational_fusion_result":
                                operational_fusion_result,

                            "operational_fusion_error":
                                operational_fusion_error,

                            # =========================================
                            # Diagnostic information
                            # =========================================

                            "mean_true_noise":
                                (
                                    sum(
                                        true_noises
                                    )
                                    / len(
                                        true_noises
                                    )
                                ),

                            "mean_estimated_noise":
                                (
                                    sum(
                                        estimated_noises
                                    )
                                    / len(
                                        estimated_noises
                                    )
                                ),

                            "mean_true_drift":
                                (
                                    sum(
                                        true_drifts
                                    )
                                    / len(
                                        true_drifts
                                    )
                                ),

                            "mean_estimated_drift":
                                (
                                    sum(
                                        estimated_drifts
                                    )
                                    / len(
                                        estimated_drifts
                                    )
                                ),

                            "mean_signed_drift":
                                (
                                    sum(
                                        signed_drifts
                                    )
                                    / len(
                                        signed_drifts
                                    )
                                ),

                            "mean_confidence":
                                (
                                    sum(
                                        confidences
                                    )
                                    / len(
                                        confidences
                                    )
                                ),

                            "mean_trust":
                                (
                                    sum(
                                        trusts
                                    )
                                    / len(
                                        trusts
                                    )
                                ),

                            "mean_backend_quality":
                                (
                                    sum(
                                        backend_qualities
                                    )
                                    / len(
                                        backend_qualities
                                    )
                                ),

                            "mean_staleness_error":
                                (
                                    sum(
                                        staleness_errors
                                    )
                                    / len(
                                        staleness_errors
                                    )
                                ),

                            "max_staleness_error":
                                max(
                                    staleness_errors
                                ),

                            "sigma_floor":
                                (
                                    fusion_engine
                                    .config
                                    .sigma_floor
                                ),
                        }
                    )

    return (
        rows
    )


# ---------------------------------------------------------------------
# Console summary
# ---------------------------------------------------------------------

def print_summary(
    rows: list[dict],
    seed: int,
) -> None:
    """
    Print summary information for one fog-fusion run.
    """

    total_runs = len(
        rows
    )

    fusion_allowed_runs = sum(

        1

        for row
        in rows

        if row[
            "can_fuse"
        ]
    )

    fusion_blocked_runs = (

        total_runs

        - fusion_allowed_runs
    )

    print()

    print(
        "Fog fusion evaluation completed."
    )

    print(
        f"Random seed: "
        f"{seed}"
    )

    print()

    print(
        "Evaluated fusion methods:"
    )

    print(
        "  1. Simple average"
    )

    print(
        "  2. Confidence weighted"
    )

    print(
        "  3. Uncertainty weighted"
    )

    print(
        "  4. Trust only"
    )

    print(
        "  5. Median fusion"
    )

    print(
        "  6. Original Trust + Uncertainty "
        "(T / sigma^2)"
    )

    print(
        "  7. Orthogonal Reliability + Uncertainty "
        "(R / sigma^2, proposed)"
    )

    print()

    print(
        "Fog assessment:"
    )

    print(
        "  Total scenarios: "
        f"{total_runs}"
    )

    print(
        "  Fusion allowed: "
        f"{fusion_allowed_runs}"
    )

    print(
        "  Fusion blocked: "
        f"{fusion_blocked_runs}"
    )

    if total_runs > 0:

        print(
            "  Fusion allowed rate: "
            f"{fusion_allowed_runs / total_runs:.2%}"
        )


# ---------------------------------------------------------------------
# Main single-seed entry point
# ---------------------------------------------------------------------

def main() -> None:
    """
    Run the standard single-seed fog-fusion experiment.

    Normal execution uses seed=42.

    Multi-seed evaluation calls evaluate(seed=...) directly and
    therefore does not save or overwrite fog_fusion_results.csv.
    """

    seed = (
        42
    )

    rows = (
        evaluate(
            seed=seed
        )
    )

    output_path = (
        RESULTS_DIR
        / "fog_fusion_results.csv"
    )

    save_csv(
        output_path,
        rows,
    )

    print_summary(
        rows=
            rows,

        seed=
            seed,
    )

    print()

    print(
        "Saved results to: "
        f"{output_path}"
    )


if __name__ == "__main__":
    main()