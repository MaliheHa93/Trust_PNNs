from __future__ import annotations

import csv

from pathlib import Path
from types import SimpleNamespace

from core.trust_manager import (
    TrustConfig,
    TrustManager,
)


RESULTS_DIR = Path(
    "evaluation/results"
)

RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


def save_csv(
    path: Path,
    rows: list[
        dict
    ],
) -> None:

    with path.open(
        "w",
        newline="",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=list(
                rows[
                    0
                ].keys()
            ),
        )

        writer.writeheader()

        writer.writerows(
            rows
        )


def main() -> None:

    manager = TrustManager(

        TrustConfig(
            forward_on_missing_quality=True,
        )
    )

    base_invocation = SimpleNamespace(

        backend_id=(
            "missing-telemetry-pnn"
        ),

        task_id=(
            "missing-telemetry-test"
        ),

        output_payload={
            "value":
                1.0,

            "modality":
                "synthetic_numeric",
        },

        confidence=(
            0.90
        ),

        execution_latency_ms=(
            100.0
        ),

        backend_state=(
            "ready"
        ),

        notes=None,
    )

    complete_telemetry = {

        "noise_score":
            0.10,

        "drift_score":
            0.10,

        "age_of_information_ms":
            100.0,

        "output_modality":
            "synthetic_numeric",
    }

    cases = {

        "complete":
            complete_telemetry,

        "missing_noise": {
            key: value

            for key, value
            in complete_telemetry.items()

            if key
            != "noise_score"
        },

        "missing_drift": {
            key: value

            for key, value
            in complete_telemetry.items()

            if key
            != "drift_score"
        },

        "missing_freshness": {
            key: value

            for key, value
            in complete_telemetry.items()

            if key
            != "age_of_information_ms"
        },

        "missing_noise_and_drift": {

            "age_of_information_ms":
                100.0,

            "output_modality":
                "synthetic_numeric",
        },
    }

    rows: list[
        dict
    ] = []

    for (
        case_name,
        telemetry,
    ) in cases.items():

        result = (
            manager
            .evaluate_invocation(

                invocation=(
                    base_invocation
                ),

                telemetry_after=(
                    telemetry
                ),
            )
        )

        missing_fields = (
            result
            .evidence
            .provenance
            .get(
                "missing_quality_fields",
                [],
            )
        )

        rows.append(
            {
                "case":
                    case_name,

                "missing_fields":
                    ",".join(
                        missing_fields
                    ),

                "trust":
                    result
                    .evidence
                    .trust,

                "decision":
                    result
                    .decision
                    .value,

                "reason":
                    result.reason,
            }
        )

    output_path = (
        RESULTS_DIR
        / "missing_telemetry_results.csv"
    )

    save_csv(
        output_path,
        rows,
    )

    print(
        "Saved missing telemetry results to: "
        f"{output_path}"
    )


if __name__ == "__main__":
    main()