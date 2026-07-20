from __future__ import annotations

import csv
import random
from pathlib import Path
from statistics import mean, median, pstdev
from time import perf_counter
from types import SimpleNamespace

from core.fog_fusion import (
    FogFusionConfig,
    FogFusionEngine,
)
from core.trust_manager import (
    TrustConfig,
    TrustManager,
)
from core.trust_models import EvidenceRecord

from evaluation.experiment_config import (
    ExperimentConfig,
)
from evaluation.scenario_generator import (
    make_random_scenario,
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
    Save scalability results to CSV.
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
# Percentile utility
# ---------------------------------------------------------------------

def percentile(
    values: list[float],
    percentile_value: float,
) -> float:
    """
    Compute a percentile using linear interpolation.

    This avoids adding a dependency such as NumPy.
    """

    if not values:
        raise ValueError(
            "Cannot calculate percentile "
            "of an empty list."
        )

    sorted_values = sorted(
        values
    )

    if len(
        sorted_values
    ) == 1:

        return sorted_values[
            0
        ]

    index = (
        percentile_value
        / 100.0
        * (
            len(
                sorted_values
            )
            - 1
        )
    )

    lower_index = int(
        index
    )

    upper_index = min(
        lower_index + 1,
        len(
            sorted_values
        )
        - 1,
    )

    interpolation_weight = (
        index
        - lower_index
    )

    return (

        sorted_values[
            lower_index
        ]

        * (
            1.0
            - interpolation_weight
        )

        + sorted_values[
            upper_index
        ]

        * interpolation_weight
    )


# ---------------------------------------------------------------------
# Invocation construction
# ---------------------------------------------------------------------

def make_invocation_from_scenario(
    scenario,
):
    """
    Construct an invocation compatible with TrustManager.
    """

    return SimpleNamespace(

        backend_id=(
            "synthetic-pnn"
        ),

        task_id=(
            f"scenario-"
            f"{scenario.scenario_id}"
        ),

        output_payload={
            "value":
                scenario.predicted_value,

            "modality":
                "synthetic_numeric",
        },

        confidence=(
            scenario.confidence
        ),

        execution_latency_ms=(
            scenario.delay_s
            * 1000.0
        ),

        backend_state=(
            "ready"
        ),

        notes=None,
    )


# ---------------------------------------------------------------------
# Build fog-fusion records
# ---------------------------------------------------------------------

def build_fusion_records(
    source_count: int,
    cfg: ExperimentConfig,
    trust_manager: TrustManager,
) -> list[EvidenceRecord]:
    """
    Generate a fixed set of evidence records for scalability testing.

    The records are generated once per source count and then reused
    across repeated timing measurements.

    This ensures that runtime measurements are not contaminated by
    scenario generation time.
    """

    records: list[
        EvidenceRecord
    ] = []

    reference_timestamp = (
        1_000_000.0
    )

    for source_id in range(
        source_count
    ):

        scenario = (
            make_random_scenario(

                scenario_id=(
                    source_id
                ),

                true_value=(
                    cfg.true_value
                ),

                error_threshold=(
                    cfg.error_threshold
                ),

                correlated_confidence=True,
            )
        )

        freshness = (
            trust_manager
            .compute_freshness(
                scenario.delay_s
            )
        )

        trust = (
            trust_manager
            .compute_trust(

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

        timestamp = (

            reference_timestamp

            - scenario.delay_s
        )

        records.append(

            EvidenceRecord(

                backend_id=(
                    f"source-"
                    f"{source_id}"
                ),

                # All sources belong to the same task.
                task_id=(
                    "fusion-scalability-task"
                ),

                value=(
                    scenario.predicted_value
                ),

                confidence=(
                    scenario.confidence
                ),

                noise=(
                    scenario.noise
                ),

                drift=(
                    scenario.drift
                ),

                timestamp=(
                    timestamp
                ),

                modality=(
                    "synthetic_numeric"
                ),

                provenance={

                    "source_id":
                        source_id,

                    "decision_context":
                        "scalability-test",

                    "observation_window_id":
                        "scalability-window",
                },

                freshness=(
                    freshness
                ),

                trust=(
                    trust
                ),
            )
        )

    return (
        records
    )


# ---------------------------------------------------------------------
# Edge trust scalability benchmark
# ---------------------------------------------------------------------

def benchmark_edge_trust(
    cfg: ExperimentConfig,
    trust_manager: TrustManager,
) -> list[dict]:
    """
    Measure edge-level trust evaluation scalability.

    The experiment measures complete trust evaluation for batches of
    evidence records.

    Scenario generation is performed outside the timed region.
    """

    rows: list[
        dict
    ] = []

    # -------------------------------------------------------------
    # Repeated timing measurements.
    #
    # Edge experiments process large batches, so fewer repetitions
    # are sufficient than for microsecond-scale fog fusion.
    # -------------------------------------------------------------

    timing_repetitions = (
        10
    )

    warmup_repetitions = (
        2
    )

    for scenario_count in (
        cfg.scalability_scenarios
    ):

        # -------------------------------------------------------------
        # Generate scenarios once.
        #
        # Scenario generation time must not be included in trust
        # computation scalability.
        # -------------------------------------------------------------

        scenarios = [

            make_random_scenario(

                scenario_id=i,

                true_value=(
                    cfg.true_value
                ),

                error_threshold=(
                    cfg.error_threshold
                ),

                correlated_confidence=True,
            )

            for i
            in range(
                scenario_count
            )
        ]

        # -------------------------------------------------------------
        # Pre-build invocations and telemetry.
        #
        # This isolates TrustManager runtime from object construction.
        # -------------------------------------------------------------

        evaluation_inputs: list[
            tuple
        ] = []

        for scenario in scenarios:

            invocation = (
                make_invocation_from_scenario(
                    scenario
                )
            )

            telemetry = {

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
                    "synthetic_numeric",
            }

            evaluation_inputs.append(
                (
                    invocation,
                    telemetry,
                )
            )

        # -------------------------------------------------------------
        # Warm-up.
        # -------------------------------------------------------------

        for _ in range(
            warmup_repetitions
        ):

            for (
                invocation,
                telemetry,
            ) in evaluation_inputs:

                trust_manager \
                    .evaluate_invocation(

                        invocation=(
                            invocation
                        ),

                        telemetry_after=(
                            telemetry
                        ),
                    )

        # -------------------------------------------------------------
        # Repeated timing.
        # -------------------------------------------------------------

        timing_values_ms: list[
            float
        ] = []

        for _ in range(
            timing_repetitions
        ):

            start = (
                perf_counter()
            )

            for (
                invocation,
                telemetry,
            ) in evaluation_inputs:

                trust_manager \
                    .evaluate_invocation(

                        invocation=(
                            invocation
                        ),

                        telemetry_after=(
                            telemetry
                        ),
                    )

            elapsed_ms = (

                perf_counter()
                - start

            ) * 1000.0

            timing_values_ms.append(
                elapsed_ms
            )

        mean_total_ms = mean(
            timing_values_ms
        )

        median_total_ms = median(
            timing_values_ms
        )

        std_total_ms = pstdev(
            timing_values_ms
        )

        p95_total_ms = percentile(
            timing_values_ms,
            95.0,
        )

        rows.append(
            {
                "experiment":
                    "edge_trust_scalability",

                "scenario_count":
                    scenario_count,

                "source_count":
                    None,

                "timing_repetitions":
                    timing_repetitions,

                "mean_total_time_ms":
                    mean_total_ms,

                "median_total_time_ms":
                    median_total_ms,

                "std_total_time_ms":
                    std_total_ms,

                "p95_total_time_ms":
                    p95_total_ms,

                "mean_time_per_record_ms":
                    (
                        mean_total_ms
                        / scenario_count
                    ),

                # Backward compatibility with old plots.
                "total_time_ms":
                    mean_total_ms,

                "time_per_record_ms":
                    (
                        mean_total_ms
                        / scenario_count
                    ),
            }
        )

        print(

            "Edge trust scalability: "

            f"{scenario_count} records | "

            f"mean={mean_total_ms:.4f} ms | "

            f"median={median_total_ms:.4f} ms | "

            f"p95={p95_total_ms:.4f} ms"
        )

    return (
        rows
    )


# ---------------------------------------------------------------------
# Fog fusion scalability benchmark
# ---------------------------------------------------------------------

def benchmark_fog_fusion(
    cfg: ExperimentConfig,
    trust_manager: TrustManager,
    fusion_engine: FogFusionEngine,
) -> list[dict]:
    """
    Measure fog-level fusion scalability.

    Two operations are measured separately:

    1. Fusion kernel:
       Numerical R_i / sigma_i^2 fusion only.

       Compatibility/disagreement enforcement is disabled because
       scalability should measure computational cost independently
       of whether a randomly generated evidence set would be allowed
       to fuse.

    2. Fog assessment:
       Compatibility and disagreement checking.

    This separation also prevents high-disagreement random evidence
    from raising an exception during a runtime benchmark.
    """

    rows: list[
        dict
    ] = []

    # -------------------------------------------------------------
    # Fusion is extremely fast, so thousands of repetitions are
    # required for stable timing measurements.
    # -------------------------------------------------------------

    timing_repetitions = (
        10_000
    )

    warmup_repetitions = (
        1_000
    )

    for source_count in (
        cfg.source_counts
    ):

        records = (
            build_fusion_records(

                source_count=(
                    source_count
                ),

                cfg=(
                    cfg
                ),

                trust_manager=(
                    trust_manager
                ),
            )
        )

        # =========================================================
        # PART A:
        # Numerical fusion-kernel scalability
        # =========================================================

        # ---------------------------------------------------------
        # Warm-up
        # ---------------------------------------------------------

        for _ in range(
            warmup_repetitions
        ):

            fusion_engine \
                .fuse_numeric(

                    records,

                    # IMPORTANT:
                    #
                    # The scalability benchmark measures the
                    # numerical fusion kernel independently from
                    # the operational decision policy.
                    enforce_assessment=False,
                )

        # ---------------------------------------------------------
        # Repeated individual timings
        # ---------------------------------------------------------

        fusion_times_ms: list[
            float
        ] = []

        for _ in range(
            timing_repetitions
        ):

            start = (
                perf_counter()
            )

            fusion_engine \
                .fuse_numeric(

                    records,

                    enforce_assessment=False,
                )

            elapsed_ms = (

                perf_counter()
                - start

            ) * 1000.0

            fusion_times_ms.append(
                elapsed_ms
            )

        fusion_mean_ms = mean(
            fusion_times_ms
        )

        fusion_median_ms = median(
            fusion_times_ms
        )

        fusion_std_ms = pstdev(
            fusion_times_ms
        )

        fusion_p95_ms = percentile(
            fusion_times_ms,
            95.0,
        )

        rows.append(
            {
                "experiment":
                    "fog_fusion_scalability",

                "scenario_count":
                    None,

                "source_count":
                    source_count,

                "timing_repetitions":
                    timing_repetitions,

                "mean_total_time_ms":
                    fusion_mean_ms,

                "median_total_time_ms":
                    fusion_median_ms,

                "std_total_time_ms":
                    fusion_std_ms,

                "p95_total_time_ms":
                    fusion_p95_ms,

                "mean_time_per_record_ms":
                    (
                        fusion_mean_ms
                        / source_count
                    ),

                # Backward-compatible columns.
                "total_time_ms":
                    fusion_mean_ms,

                "time_per_record_ms":
                    (
                        fusion_mean_ms
                        / source_count
                    ),
            }
        )

        # =========================================================
        # PART B:
        # Compatibility + disagreement assessment scalability
        # =========================================================

        for _ in range(
            warmup_repetitions
        ):

            fusion_engine.assess(
                records
            )

        assessment_times_ms: list[
            float
        ] = []

        for _ in range(
            timing_repetitions
        ):

            start = (
                perf_counter()
            )

            fusion_engine.assess(
                records
            )

            elapsed_ms = (

                perf_counter()
                - start

            ) * 1000.0

            assessment_times_ms.append(
                elapsed_ms
            )

        assessment_mean_ms = mean(
            assessment_times_ms
        )

        assessment_median_ms = median(
            assessment_times_ms
        )

        assessment_std_ms = pstdev(
            assessment_times_ms
        )

        assessment_p95_ms = percentile(
            assessment_times_ms,
            95.0,
        )

        rows.append(
            {
                "experiment":
                    "fog_assessment_scalability",

                "scenario_count":
                    None,

                "source_count":
                    source_count,

                "timing_repetitions":
                    timing_repetitions,

                "mean_total_time_ms":
                    assessment_mean_ms,

                "median_total_time_ms":
                    assessment_median_ms,

                "std_total_time_ms":
                    assessment_std_ms,

                "p95_total_time_ms":
                    assessment_p95_ms,

                "mean_time_per_record_ms":
                    (
                        assessment_mean_ms
                        / source_count
                    ),

                "total_time_ms":
                    assessment_mean_ms,

                "time_per_record_ms":
                    (
                        assessment_mean_ms
                        / source_count
                    ),
            }
        )

        print(

            "Fog scalability: "

            f"{source_count} sources | "

            f"fusion mean="
            f"{fusion_mean_ms:.6f} ms | "

            f"assessment mean="
            f"{assessment_mean_ms:.6f} ms"
        )

    return (
        rows
    )


# ---------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------

def main() -> None:

    cfg = (
        ExperimentConfig()
    )

    random.seed(
        cfg.seed
    )

    # -----------------------------------------------------------------
    # Trust manager
    # -----------------------------------------------------------------

    trust_manager = (

        TrustManager(

            TrustConfig(

                alpha=(
                    cfg.alpha
                ),

                beta=(
                    cfg.beta
                ),

                gamma=(
                    cfg.gamma
                ),

                delta=(
                    cfg.delta
                ),

                tau_s=(
                    cfg.tau_s
                ),

                theta_accept=(
                    cfg.theta_accept
                ),

                theta_reject=(
                    cfg.theta_reject
                ),
            )
        )
    )

    # -----------------------------------------------------------------
    # Fog fusion engine
    # -----------------------------------------------------------------

    fusion_engine = (

        FogFusionEngine(

            FogFusionConfig(

                sigma_floor=(
                    0.02
                ),

                epsilon=(
                    1e-6
                ),

                max_timestamp_difference_s=(
                    20.0
                ),

                max_normalized_disagreement=(
                    0.75
                ),

                confidence_weight=(
                    1.0
                    / 3.0
                ),

                drift_weight=(
                    1.0
                    / 3.0
                ),

                freshness_weight=(
                    1.0
                    / 3.0
                ),
            )
        )
    )

    rows: list[
        dict
    ] = []

    # -----------------------------------------------------------------
    # Edge trust scalability
    # -----------------------------------------------------------------

    print()

    print(
        "Running edge trust scalability benchmark..."
    )

    edge_rows = (
        benchmark_edge_trust(

            cfg=(
                cfg
            ),

            trust_manager=(
                trust_manager
            ),
        )
    )

    rows.extend(
        edge_rows
    )

    # -----------------------------------------------------------------
    # Fog fusion scalability
    # -----------------------------------------------------------------

    print()

    print(
        "Running fog fusion scalability benchmark..."
    )

    fog_rows = (
        benchmark_fog_fusion(

            cfg=(
                cfg
            ),

            trust_manager=(
                trust_manager
            ),

            fusion_engine=(
                fusion_engine
            ),
        )
    )

    rows.extend(
        fog_rows
    )

    # -----------------------------------------------------------------
    # Save results
    # -----------------------------------------------------------------

    output_path = (

        RESULTS_DIR

        / "scalability_results.csv"
    )

    save_csv(
        output_path,
        rows,
    )

    print()

    print(
        "Scalability evaluation completed."
    )

    print(
        "Saved scalability results to: "
        f"{output_path}"
    )

    print()

    print(
        "Scalability experiments:"
    )

    print(
        "  1. Edge trust evaluation"
    )

    print(
        "  2. Fog numerical fusion kernel"
    )

    print(
        "  3. Fog compatibility/disagreement assessment"
    )


# ---------------------------------------------------------------------
# Script entry point
# ---------------------------------------------------------------------

if __name__ == "__main__":
    main()