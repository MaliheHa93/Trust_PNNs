from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from core.fog_fusion import FogFusionEngine
from core.trust_models import EvidenceRecord


RESULTS_DIR = Path("evaluation/results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_FILE = RESULTS_DIR / "fog_compatibility_results.csv"


def make_record(
    *,
    backend_id: str,
    task_id: str = "task-001",
    value: float = 1.0,
    confidence: float = 0.8,
    noise: float = 0.1,
    drift: float = 0.1,
    timestamp: float = 100.0,
    modality: str = "chemical_concentration",
    decision_context: str = "context-A",
    observation_window_id: str = "window-A",
) -> EvidenceRecord:
    """
    Construct a controlled evidence record for fog-level
    compatibility validation.
    """

    return EvidenceRecord(
        backend_id=backend_id,
        task_id=task_id,
        value=value,
        confidence=confidence,
        noise=noise,
        drift=drift,
        timestamp=timestamp,
        modality=modality,
        provenance={
            "decision_context": decision_context,
            "observation_window_id": observation_window_id,
        },
        freshness=0.9,
        trust=0.8,
    )


def get_compatibility(
    engine: FogFusionEngine,
    records: list[EvidenceRecord],
) -> tuple[bool, str]:
    """
    Evaluate compatibility using the FogFusionEngine.

    Supports both:
      - assess()
      - are_compatible()
    """

    if hasattr(engine, "assess"):
        assessment = engine.assess(records)

        # Dictionary-style assessment
        if isinstance(assessment, dict):
            compatible = bool(
                assessment.get(
                    "compatible",
                    assessment.get(
                        "is_compatible",
                        False,
                    ),
                )
            )

            reason = str(
                assessment.get(
                    "compatibility_reason",
                    assessment.get(
                        "reason",
                        "",
                    ),
                )
            )

            return compatible, reason

        # Dataclass/object-style assessment
        compatible = bool(
            getattr(
                assessment,
                "compatible",
                getattr(
                    assessment,
                    "is_compatible",
                    False,
                ),
            )
        )

        reason = str(
            getattr(
                assessment,
                "compatibility_reason",
                getattr(
                    assessment,
                    "reason",
                    "",
                ),
            )
        )

        return compatible, reason

    compatible = engine.are_compatible(records)

    return (
        compatible,
        "Compatibility evaluated using are_compatible().",
    )


def build_test_cases() -> list[dict[str, Any]]:
    """
    Construct deterministic compatibility test cases.

    Each incompatible test modifies one compatibility condition
    while keeping the remaining conditions unchanged.
    """

    base_record = make_record(
        backend_id="backend-A",
    )

    return [

        # -----------------------------------------------------
        # 1. Fully compatible evidence
        # -----------------------------------------------------
        {
            "test_case": "fully_compatible",
            "description": (
                "Same task, modality, temporal window, "
                "decision context, and observation window."
            ),
            "records": [
                base_record,
                make_record(
                    backend_id="backend-B",
                    value=1.05,
                    timestamp=101.0,
                ),
            ],
            "expected_compatible": True,
        },

        # -----------------------------------------------------
        # 2. Task mismatch
        # -----------------------------------------------------
        {
            "test_case": "task_id_mismatch",
            "description": (
                "Evidence records refer to different task identifiers."
            ),
            "records": [
                base_record,
                make_record(
                    backend_id="backend-B",
                    task_id="task-002",
                ),
            ],
            "expected_compatible": False,
        },

        # -----------------------------------------------------
        # 3. Modality mismatch
        # -----------------------------------------------------
        {
            "test_case": "modality_mismatch",
            "description": (
                "Evidence records use different output modalities."
            ),
            "records": [
                base_record,
                make_record(
                    backend_id="backend-B",
                    modality="optical_intensity",
                ),
            ],
            "expected_compatible": False,
        },

        # -----------------------------------------------------
        # 4. Decision-context mismatch
        # -----------------------------------------------------
        {
            "test_case": "decision_context_mismatch",
            "description": (
                "Evidence records refer to different "
                "decision contexts."
            ),
            "records": [
                base_record,
                make_record(
                    backend_id="backend-B",
                    decision_context="context-B",
                ),
            ],
            "expected_compatible": False,
        },

        # -----------------------------------------------------
        # 5. Observation-window mismatch
        # -----------------------------------------------------
        {
            "test_case": "observation_window_mismatch",
            "description": (
                "Evidence records belong to different "
                "observation windows."
            ),
            "records": [
                base_record,
                make_record(
                    backend_id="backend-B",
                    observation_window_id="window-B",
                ),
            ],
            "expected_compatible": False,
        },

        # -----------------------------------------------------
        # 6. Timestamp mismatch
        # -----------------------------------------------------
        {
            "test_case": "timestamp_mismatch",
            "description": (
                "Evidence timestamps exceed the configured "
                "compatibility window."
            ),
            "records": [
                base_record,
                make_record(
                    backend_id="backend-B",
                    timestamp=1100.0,
                ),
            ],
            "expected_compatible": False,
        },
    ]


def evaluate() -> list[dict[str, Any]]:
    """
    Evaluate all fog-level compatibility conditions.
    """

    engine = FogFusionEngine()

    test_cases = build_test_cases()

    rows: list[dict[str, Any]] = []

    for test_case in test_cases:

        actual_compatible, reason = get_compatibility(
            engine=engine,
            records=test_case["records"],
        )

        expected_compatible = bool(
            test_case["expected_compatible"]
        )

        passed = (
            actual_compatible
            == expected_compatible
        )

        rows.append(
            {
                "test_case":
                    test_case["test_case"],

                "description":
                    test_case["description"],

                "expected_compatible":
                    expected_compatible,

                "actual_compatible":
                    actual_compatible,

                "passed":
                    passed,

                "reason":
                    reason,
            }
        )

    return rows


def save_csv(
    path: Path,
    rows: list[dict[str, Any]],
) -> None:
    """
    Save compatibility-validation results.
    """

    if not rows:
        raise ValueError(
            "No fog compatibility results to save."
        )

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
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


def print_summary(
    rows: list[dict[str, Any]],
) -> None:
    """
    Print fog-level compatibility validation results.
    """

    print(
        "\nFog Compatibility Validation"
    )

    print(
        "=" * 85
    )

    print(
        f"{'Test Case':<32}"
        f"{'Expected':>12}"
        f"{'Actual':>12}"
        f"{'Passed':>12}"
    )

    print(
        "-" * 85
    )

    for row in rows:

        print(
            f"{row['test_case']:<32}"
            f"{str(row['expected_compatible']):>12}"
            f"{str(row['actual_compatible']):>12}"
            f"{str(row['passed']):>12}"
        )

    passed_count = sum(
        1
        for row in rows
        if row["passed"]
    )

    total_count = len(rows)

    print(
        "-" * 85
    )

    print(
        f"Passed {passed_count}/{total_count} "
        f"compatibility tests."
    )


def main() -> None:
    """
    Run fog-level compatibility validation.
    """

    rows = evaluate()

    save_csv(
        OUTPUT_FILE,
        rows,
    )

    print_summary(
        rows,
    )

    print(
        f"\nSaved fog compatibility results to: "
        f"{OUTPUT_FILE}"
    )


if __name__ == "__main__":
    main()