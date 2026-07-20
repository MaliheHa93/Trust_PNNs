from __future__ import annotations

from evaluation import evaluate_trust_baselines
from evaluation import evaluate_risk_coverage
from evaluation import evaluate_fog_fusion
from evaluation import evaluate_fog_compatibility
from evaluation import evaluate_random_scenarios
from evaluation import evaluate_ablation
from evaluation import evaluate_sensitivity
from evaluation import evaluate_adaptive_trust
from evaluation import evaluate_missing_telemetry
from evaluation import evaluate_scalability
from evaluation import evaluate_trust_overhead


def main() -> None:
    """
    Run the complete trust-aware PNN evaluation suite.

    Evaluation order:
        1. Controlled trust baseline evaluation
        2. Risk--coverage analysis
        3. Fog-level fusion evaluation
        4. Fog-level compatibility validation
        5. Randomized Monte Carlo stress evaluation
        6. Trust-component ablation study
        7. Trust-weight and threshold sensitivity analysis
        8. Adaptive trust and recalibration evaluation
        9. Missing-telemetry robustness evaluation
       10. Computational scalability evaluation
       11. Trust-layer runtime overhead evaluation
    """

    total_evaluations = 11

    print()
    print("=" * 72)
    print("TRUST-AWARE PNN EVALUATION SUITE")
    print("=" * 72)

    # =============================================================
    # 1. Controlled trust baseline evaluation
    # =============================================================

    print()
    print(
        f"[1/{total_evaluations}] "
        "Running trust baseline evaluation..."
    )

    evaluate_trust_baselines.main()

    # =============================================================
    # 2. Risk--coverage evaluation
    #
    # This evaluation must run after the baseline evaluation because
    # it uses trust_baselines_details.csv.
    # =============================================================

    print()
    print(
        f"[2/{total_evaluations}] "
        "Running risk--coverage evaluation..."
    )

    evaluate_risk_coverage.main()

    # =============================================================
    # 3. Fog-level fusion evaluation
    #
    # Compares:
    #   - simple averaging
    #   - confidence-weighted averaging
    #   - uncertainty-weighted averaging
    #   - trust-only weighting
    #   - median fusion
    #   - original T / sigma^2 fusion
    #   - proposed R / sigma^2 fusion
    #
    # Also evaluates operational compatibility and disagreement gates.
    # =============================================================

    print()
    print(
        f"[3/{total_evaluations}] "
        "Running fog fusion evaluation..."
    )

    evaluate_fog_fusion.main()

    # =============================================================
    # 4. Fog-level compatibility validation
    #
    # Validates:
    #   - task identifier
    #   - modality
    #   - timestamp window
    #   - decision context
    #   - observation-window identifier
    # =============================================================

    print()
    print(
        f"[4/{total_evaluations}] "
        "Running fog compatibility evaluation..."
    )

    evaluate_fog_compatibility.main()

    # =============================================================
    # 5. Randomized Monte Carlo stress evaluation
    # =============================================================

    print()
    print(
        f"[5/{total_evaluations}] "
        "Running randomized stress-test evaluation..."
    )

    evaluate_random_scenarios.main()

    # =============================================================
    # 6. Trust-component ablation study
    #
    # Evaluates the contribution of:
    #   - confidence
    #   - uncertainty/noise
    #   - drift
    #   - freshness
    # =============================================================

    print()
    print(
        f"[6/{total_evaluations}] "
        "Running trust-component ablation evaluation..."
    )

    evaluate_ablation.main()

    # =============================================================
    # 7. Trust parameter sensitivity analysis
    #
    # Evaluates:
    #   - threshold strictness
    #   - trust-component weight configurations
    # =============================================================

    print()
    print(
        f"[7/{total_evaluations}] "
        "Running sensitivity evaluation..."
    )

    evaluate_sensitivity.main()

    # =============================================================
    # 8. Adaptive trust and recalibration evaluation
    #
    # Evaluates:
    #   - current trust
    #   - historical trust
    #   - repeated low-trust behavior
    #   - recalibration trigger
    #   - recovery behavior
    # =============================================================

    print()
    print(
        f"[8/{total_evaluations}] "
        "Running adaptive trust evaluation..."
    )

    evaluate_adaptive_trust.main()

    # =============================================================
    # 9. Missing-telemetry robustness evaluation
    #
    # Verifies the conservative forwarding policy when reliability
    # indicators are incomplete.
    # =============================================================

    print()
    print(
        f"[9/{total_evaluations}] "
        "Running missing-telemetry evaluation..."
    )

    evaluate_missing_telemetry.main()

    # =============================================================
    # 10. Computational scalability evaluation
    #
    # Separates:
    #   - edge trust computation
    #   - fog numerical fusion
    #   - fog compatibility/disagreement assessment
    # =============================================================

    print()
    print(
        f"[10/{total_evaluations}] "
        "Running scalability evaluation..."
    )

    evaluate_scalability.main()

    # =============================================================
    # 11. Trust-layer runtime overhead evaluation
    # =============================================================

    print()
    print(
        f"[11/{total_evaluations}] "
        "Running trust-layer overhead evaluation..."
    )

    evaluate_trust_overhead.main()

    # =============================================================
    # Completion
    # =============================================================

    print()
    print("=" * 72)
    print(
        "ALL TRUST-AWARE PNN EVALUATIONS "
        "COMPLETED SUCCESSFULLY"
    )
    print("=" * 72)
    print()


if __name__ == "__main__":
    main()