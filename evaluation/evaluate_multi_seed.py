from __future__ import annotations

import csv
import math
from pathlib import Path
from statistics import mean, stdev

from evaluation.evaluate_trust_baselines import (
    evaluate as evaluate_trust_baselines,
)
from evaluation.evaluate_random_scenarios import (
    evaluate as evaluate_random_scenarios,
)
from evaluation.evaluate_fog_fusion import (
    evaluate as evaluate_fog_fusion,
)


RESULTS_DIR = Path("evaluation/results")
RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ---------------------------------------------------------------------
# Independent random seeds
#
# Define the seed list ONLY here.
# ---------------------------------------------------------------------

SEEDS = [
    42,
    101,
    202,
    303,
    404,
    505,
    606,
    707,
    808,
    909,
    1010,
    1111,
    1212,
    1313,
    1414,
    1515,
    1616,
    1717,
    1818,
    1919,
]


# Main controlled-baseline configuration used in the paper.
BASELINE_TAU_S = 5.0


# ---------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------

def save_csv(
    path: Path,
    rows: list[dict],
) -> None:

    if not rows:
        raise ValueError(
            f"No rows available for {path}."
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


def as_bool(
    value,
) -> bool:

    if isinstance(
        value,
        bool,
    ):
        return value

    return (
        str(value)
        .strip()
        .lower()
        in {
            "true",
            "1",
            "yes",
        }
    )


# ---------------------------------------------------------------------
# 95% Student-t critical value
#
# For 20 seeds:
# df = 19
# t_0.975 ≈ 2.093
#
# Values for smaller sample sizes are included for robustness.
# ---------------------------------------------------------------------

def t_critical_95(
    n: int,
) -> float:

    if n < 2:
        return 0.0

    table = {
        1: 12.706,
        2: 4.303,
        3: 3.182,
        4: 2.776,
        5: 2.571,
        6: 2.447,
        7: 2.365,
        8: 2.306,
        9: 2.262,
        10: 2.228,
        11: 2.201,
        12: 2.179,
        13: 2.160,
        14: 2.145,
        15: 2.131,
        16: 2.120,
        17: 2.110,
        18: 2.101,
        19: 2.093,
        20: 2.086,
        21: 2.080,
        22: 2.074,
        23: 2.069,
        24: 2.064,
        25: 2.060,
        26: 2.056,
        27: 2.052,
        28: 2.048,
        29: 2.045,
        30: 2.042,
    }

    df = (
        n
        - 1
    )

    return table.get(
        df,
        1.96,
    )


def confidence_interval(
    values: list[float],
) -> tuple[
    float,
    float,
    float,
    float,
]:

    if not values:
        raise ValueError(
            "Cannot calculate statistics "
            "for an empty list."
        )

    n = len(
        values
    )

    mean_value = mean(
        values
    )

    if n == 1:

        return (
            mean_value,
            0.0,
            mean_value,
            mean_value,
        )

    std_value = stdev(
        values
    )

    standard_error = (
        std_value
        / math.sqrt(
            n
        )
    )

    margin = (
        t_critical_95(
            n
        )
        * standard_error
    )

    return (
        mean_value,
        std_value,
        mean_value
        - margin,
        mean_value
        + margin,
    )


# ---------------------------------------------------------------------
# Decision-method summarization
#
# Used for:
#   - controlled baseline
#   - randomized stress test
# ---------------------------------------------------------------------

def summarize_decision_method(
    rows: list[dict],
    method: str,
) -> dict:

    items = [
        row
        for row in rows
        if row[
            "method"
        ]
        == method
    ]

    if not items:
        raise ValueError(
            f"No rows found for method: {method}"
        )

    total = len(
        items
    )

    accepted = [
        row
        for row in items
        if as_bool(
            row[
                "accepted"
            ]
        )
    ]

    rejected = [
        row
        for row in items
        if as_bool(
            row[
                "rejected"
            ]
        )
    ]

    forwarded = [
        row
        for row in items
        if as_bool(
            row[
                "forwarded"
            ]
        )
    ]

    bad_outputs = [
        row
        for row in items
        if as_bool(
            row[
                "bad_output"
            ]
        )
    ]

    accepted_bad = [
        row
        for row in accepted
        if as_bool(
            row[
                "bad_output"
            ]
        )
    ]

    accepted_errors = [
        float(
            row[
                "absolute_error"
            ]
        )
        for row in accepted
    ]

    coverage = (
        len(
            accepted
        )
        / total
    )

    forwarding_rate = (
        len(
            forwarded
        )
        / total
    )

    rejection_rate = (
        len(
            rejected
        )
        / total
    )

    # P(Accept | Bad)
    false_acceptance_rate = (
        len(
            accepted_bad
        )
        / len(
            bad_outputs
        )

        if bad_outputs
        else 0.0
    )

    # P(Accept AND Bad)
    unsafe_acceptance_frequency = (
        len(
            accepted_bad
        )
        / total
    )

    # P(Bad | Accept)
    accepted_output_contamination_rate = (
        len(
            accepted_bad
        )
        / len(
            accepted
        )

        if accepted
        else None
    )

    accepted_mae = (
        mean(
            accepted_errors
        )

        if accepted_errors
        else None
    )

    return {
        "coverage":
            coverage,

        "forwarding_rate":
            forwarding_rate,

        "rejection_rate":
            rejection_rate,

        "false_acceptance_rate":
            false_acceptance_rate,

        "unsafe_acceptance_frequency":
            unsafe_acceptance_frequency,

        "accepted_output_contamination_rate":
            accepted_output_contamination_rate,

        "accepted_mae":
            accepted_mae,
    }


# ---------------------------------------------------------------------
# Controlled baseline summarization
# ---------------------------------------------------------------------

def summarize_baseline(
    seed: int,
    rows: list[dict],
) -> list[dict]:

    # Use tau = 5 s for the main statistical comparison.
    filtered_rows = [
        row
        for row in rows
        if abs(
            float(
                row[
                    "tau_s"
                ]
            )
            - BASELINE_TAU_S
        )
        < 1e-9
    ]

    methods = [
        "raw_phys_mcp_output",
        "confidence_only",
        "freshness_only",
        "full_trust_score",
    ]

    result_rows: list[
        dict
    ] = []

    for method in methods:

        metrics = (
            summarize_decision_method(
                filtered_rows,
                method,
            )
        )

        result_rows.append(
            {
                "seed":
                    seed,

                "experiment":
                    "controlled_baseline",

                "method":
                    method,

                **metrics,

                "fusion_mae":
                    None,

                "fusion_allowed_rate":
                    None,
            }
        )

    return result_rows


# ---------------------------------------------------------------------
# Randomized stress-test summarization
# ---------------------------------------------------------------------

def summarize_random(
    seed: int,
    rows: list[dict],
) -> list[dict]:

    methods = [
        "raw_phys_mcp_output",
        "confidence_only",
        "freshness_only",
        "full_trust_score",
    ]

    result_rows: list[
        dict
    ] = []

    for method in methods:

        metrics = (
            summarize_decision_method(
                rows,
                method,
            )
        )

        result_rows.append(
            {
                "seed":
                    seed,

                "experiment":
                    "random_stress",

                "method":
                    method,

                **metrics,

                "fusion_mae":
                    None,

                "fusion_allowed_rate":
                    None,
            }
        )

    return result_rows


# ---------------------------------------------------------------------
# Fog-fusion summarization
# ---------------------------------------------------------------------

def summarize_fog(
    seed: int,
    rows: list[dict],
) -> list[dict]:

    if not rows:
        raise ValueError(
            "Fog fusion evaluation returned no rows."
        )

    method_columns = {
        "simple_average":
            "simple_average_error",

        "confidence_weighted":
            "confidence_weighted_error",

        "uncertainty_weighted":
            "uncertainty_weighted_error",

        "trust_only":
            "trust_only_error",

        "median":
            "median_error",

        "original_trust_uncertainty":
            "original_trust_uncertainty_error",

        "orthogonal_trust_uncertainty":
            "orthogonal_trust_uncertainty_error",
    }

    fusion_allowed_rate = (
        sum(
            1
            for row in rows
            if as_bool(
                row[
                    "can_fuse"
                ]
            )
        )
        / len(
            rows
        )
    )

    result_rows: list[
        dict
    ] = []

    for (
        method,
        error_column,
    ) in method_columns.items():

        fusion_mae = mean(
            [
                float(
                    row[
                        error_column
                    ]
                )
                for row in rows
            ]
        )

        result_rows.append(
            {
                "seed":
                    seed,

                "experiment":
                    "fog_fusion",

                "method":
                    method,

                "coverage":
                    None,

                "forwarding_rate":
                    None,

                "rejection_rate":
                    None,

                "false_acceptance_rate":
                    None,

                "unsafe_acceptance_frequency":
                    None,

                "accepted_output_contamination_rate":
                    None,

                "accepted_mae":
                    None,

                "fusion_mae":
                    fusion_mae,

                # The gate is common to the same scenario set.
                "fusion_allowed_rate":
                    fusion_allowed_rate,
            }
        )

    return result_rows


# ---------------------------------------------------------------------
# Aggregate statistics across seeds
# ---------------------------------------------------------------------

def build_summary(
    details: list[dict],
) -> list[dict]:

    metric_names = [
        "coverage",
        "forwarding_rate",
        "rejection_rate",
        "false_acceptance_rate",
        "unsafe_acceptance_frequency",
        "accepted_output_contamination_rate",
        "accepted_mae",
        "fusion_mae",
        "fusion_allowed_rate",
    ]

    groups: dict[
        tuple[
            str,
            str,
        ],
        list[dict],
    ] = {}

    for row in details:

        key = (
            row[
                "experiment"
            ],
            row[
                "method"
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

    for (
        experiment,
        method,
    ), items in groups.items():

        for metric in metric_names:

            values = [
                float(
                    item[
                        metric
                    ]
                )
                for item in items

                if item[
                    metric
                ]
                is not None
            ]

            if not values:
                continue

            (
                mean_value,
                std_value,
                ci_lower,
                ci_upper,
            ) = confidence_interval(
                values
            )

            summary_rows.append(
                {
                    "experiment":
                        experiment,

                    "method":
                        method,

                    "metric":
                        metric,

                    "n_seeds":
                        len(
                            values
                        ),

                    "mean":
                        mean_value,

                    "std":
                        std_value,

                    "ci95_lower":
                        ci_lower,

                    "ci95_upper":
                        ci_upper,
                }
            )

    return summary_rows


# ---------------------------------------------------------------------
# Paired comparisons
# ---------------------------------------------------------------------

def paired_comparison(
    details: list[dict],
    experiment: str,
    proposed_method: str,
    baseline_method: str,
    metric: str,
) -> dict:

    proposed_by_seed = {
        int(
            row[
                "seed"
            ]
        ):
        float(
            row[
                metric
            ]
        )

        for row in details

        if (
            row[
                "experiment"
            ]
            == experiment

            and row[
                "method"
            ]
            == proposed_method

            and row[
                metric
            ]
            is not None
        )
    }

    baseline_by_seed = {
        int(
            row[
                "seed"
            ]
        ):
        float(
            row[
                metric
            ]
        )

        for row in details

        if (
            row[
                "experiment"
            ]
            == experiment

            and row[
                "method"
            ]
            == baseline_method

            and row[
                metric
            ]
            is not None
        )
    }

    common_seeds = sorted(
        set(
            proposed_by_seed
        )
        & set(
            baseline_by_seed
        )
    )

    if not common_seeds:
        raise ValueError(
            "No paired seeds available for "
            f"{experiment}, {metric}."
        )

    # Proposed - baseline.
    #
    # For error metrics and FAR:
    # negative difference means the proposed method is better.
    differences = [
        proposed_by_seed[
            seed
        ]
        - baseline_by_seed[
            seed
        ]

        for seed in common_seeds
    ]

    (
        mean_difference,
        std_difference,
        ci_lower,
        ci_upper,
    ) = confidence_interval(
        differences
    )

    return {
        "experiment":
            experiment,

        "proposed_method":
            proposed_method,

        "baseline_method":
            baseline_method,

        "metric":
            metric,

        "n_seeds":
            len(
                common_seeds
            ),

        "mean_difference":
            mean_difference,

        "std_difference":
            std_difference,

        "ci95_lower":
            ci_lower,

        "ci95_upper":
            ci_upper,

        "ci_excludes_zero":
            (
                ci_lower > 0.0
                or ci_upper < 0.0
            ),
    }


def build_paired_comparisons(
    details: list[dict],
) -> list[dict]:

    comparisons = [

        # Controlled baseline
        (
            "controlled_baseline",
            "full_trust_score",
            "confidence_only",
            "false_acceptance_rate",
        ),

        (
            "controlled_baseline",
            "full_trust_score",
            "confidence_only",
            "accepted_mae",
        ),

        # Randomized stress test
        (
            "random_stress",
            "full_trust_score",
            "confidence_only",
            "false_acceptance_rate",
        ),

        (
            "random_stress",
            "full_trust_score",
            "confidence_only",
            "accepted_mae",
        ),

        # Fog fusion
        (
            "fog_fusion",
            "orthogonal_trust_uncertainty",
            "simple_average",
            "fusion_mae",
        ),

        (
            "fog_fusion",
            "orthogonal_trust_uncertainty",
            "confidence_weighted",
            "fusion_mae",
        ),

        (
            "fog_fusion",
            "orthogonal_trust_uncertainty",
            "uncertainty_weighted",
            "fusion_mae",
        ),

        (
            "fog_fusion",
            "orthogonal_trust_uncertainty",
            "original_trust_uncertainty",
            "fusion_mae",
        ),
    ]

    rows: list[
        dict
    ] = []

    for comparison in comparisons:

        rows.append(
            paired_comparison(
                details=details,
                experiment=comparison[
                    0
                ],
                proposed_method=comparison[
                    1
                ],
                baseline_method=comparison[
                    2
                ],
                metric=comparison[
                    3
                ],
            )
        )

    return rows


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main() -> None:

    details: list[
        dict
    ] = []

    print()
    print(
        "=" * 72
    )

    print(
        "MULTI-SEED STATISTICAL EVALUATION"
    )

    print(
        "=" * 72
    )

    for index, seed in enumerate(
        SEEDS,
        start=1,
    ):

        print()
        print(
            f"[{index}/{len(SEEDS)}] "
            f"Running seed {seed}..."
        )

        # ---------------------------------------------------------
        # Controlled baseline
        # ---------------------------------------------------------

        baseline_rows = (
            evaluate_trust_baselines(
                seed=seed
            )
        )

        details.extend(
            summarize_baseline(
                seed=seed,
                rows=baseline_rows,
            )
        )

        # ---------------------------------------------------------
        # Randomized stress test
        # ---------------------------------------------------------

        random_rows = (
            evaluate_random_scenarios(
                seed=seed
            )
        )

        details.extend(
            summarize_random(
                seed=seed,
                rows=random_rows,
            )
        )

        # ---------------------------------------------------------
        # Fog fusion
        # ---------------------------------------------------------

        fog_rows = (
            evaluate_fog_fusion(
                seed=seed
            )
        )

        details.extend(
            summarize_fog(
                seed=seed,
                rows=fog_rows,
            )
        )

    # -----------------------------------------------------------------
    # Aggregate results
    # -----------------------------------------------------------------

    summary = (
        build_summary(
            details
        )
    )

    paired = (
        build_paired_comparisons(
            details
        )
    )

    # -----------------------------------------------------------------
    # Save files
    # -----------------------------------------------------------------

    details_path = (
        RESULTS_DIR
        / "multi_seed_details.csv"
    )

    summary_path = (
        RESULTS_DIR
        / "multi_seed_summary.csv"
    )

    paired_path = (
        RESULTS_DIR
        / "multi_seed_paired_comparisons.csv"
    )

    save_csv(
        details_path,
        details,
    )

    save_csv(
        summary_path,
        summary,
    )

    save_csv(
        paired_path,
        paired,
    )

    print()
    print(
        "=" * 72
    )

    print(
        "MULTI-SEED EVALUATION COMPLETED"
    )

    print(
        "=" * 72
    )

    print(
        f"Details: {details_path}"
    )

    print(
        f"Summary: {summary_path}"
    )

    print(
        f"Paired comparisons: {paired_path}"
    )


if __name__ == "__main__":
    main()