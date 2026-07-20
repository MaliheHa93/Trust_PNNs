from __future__ import annotations

import random
from dataclasses import dataclass


@dataclass
class Scenario:
    scenario_id: int

    # Current ground-truth state at decision time.
    true_value: float

    # Physical output returned by the PNN.
    predicted_value: float

    # Confidence visible to the trust layer.
    confidence: float

    # Hidden physical uncertainty.
    true_noise: float
    true_drift: float

    # Signed physical drift used to generate the output.
    signed_drift: float

    # Imperfect telemetry estimates visible to TrustManager.
    noise: float
    drift: float

    # Age of the returned output.
    delay_s: float

    # Hidden substrate-quality factor used partly for confidence.
    backend_quality: float

    # Rate at which the underlying physical state changes.
    state_velocity: float

    # State value when the physical output was generated.
    value_at_generation: float

    # Difference caused only by stale information.
    staleness_error: float

    absolute_error: float
    bad_output: bool


def clip01(value: float) -> float:
    return max(
        0.0,
        min(
            1.0,
            float(value),
        ),
    )


def estimate_quality_indicator(
    true_value: float,
    estimation_std: float = 0.05,
) -> float:
    """
    Generate imperfect telemetry from a hidden physical condition.
    """

    return clip01(
        true_value
        + random.gauss(
            0.0,
            estimation_std,
        )
    )


def estimate_confidence(
    backend_quality: float,
    estimated_noise: float,
    estimated_drift: float,
    confidence_error_std: float = 0.05,
) -> float:
    """
    Generate confidence from multiple factors.

    Confidence is primarily influenced by an independent latent
    substrate-quality factor, while noise and drift contribute only
    partially.

    This avoids making confidence a duplicate representation of
    noise and drift.
    """

    confidence = (
        0.60 * backend_quality
        + 0.20 * (1.0 - estimated_noise)
        + 0.20 * (1.0 - estimated_drift)
        + random.gauss(
            0.0,
            confidence_error_std,
        )
    )

    return clip01(
        confidence
    )


def generate_signed_drift(
    drift_magnitude: float,
) -> float:
    """
    Physical drift may move in either direction.
    """

    direction = random.choice(
        [-1.0, 1.0]
    )

    return (
        clip01(
            drift_magnitude
        )
        * direction
    )


def generate_state_velocity(
    max_abs_rate_per_s: float = 0.02,
) -> float:
    """
    Simulate temporal evolution of the underlying physical state.

    At maximum delay of 20 s and rate 0.02 units/s, staleness can
    contribute up to approximately 0.4 units of output error.
    """

    return random.uniform(
        -max_abs_rate_per_s,
        max_abs_rate_per_s,
    )


def make_controlled_scenario(
    scenario_id: int,
    true_value: float,
    noise: float,
    drift: float,
    delay_s: float,
    error_threshold: float,
    telemetry_estimation_std: float = 0.05,
    confidence_error_std: float = 0.05,
    max_state_change_rate_per_s: float = 0.02,
) -> Scenario:
    """
    Generate one controlled physical PNN scenario.

    true_value represents the state at decision time.

    The PNN result was produced delay_s seconds earlier, when the
    physical state may have been different. Consequently, stale
    outputs can genuinely become less accurate.
    """

    true_noise = clip01(
        noise
    )

    true_drift = clip01(
        drift
    )

    signed_drift = (
        generate_signed_drift(
            true_drift
        )
    )

    # Independent substrate quality.
    backend_quality = random.uniform(
        0.30,
        1.0,
    )

    # Temporal evolution of the physical state.
    state_velocity = (
        generate_state_velocity(
            max_state_change_rate_per_s
        )
    )

    # State when the output was originally produced.
    value_at_generation = (
        true_value
        - state_velocity
        * delay_s
    )

    staleness_error = abs(
        true_value
        - value_at_generation
    )

    # Physical output generation.
    random_noise = random.gauss(
        0.0,
        true_noise,
    )

    predicted_value = (
        value_at_generation
        + random_noise
        + signed_drift
    )

    absolute_error = abs(
        predicted_value
        - true_value
    )

    # Imperfect telemetry.
    estimated_noise = (
        estimate_quality_indicator(
            true_noise,
            telemetry_estimation_std,
        )
    )

    estimated_drift = (
        estimate_quality_indicator(
            true_drift,
            telemetry_estimation_std,
        )
    )

    confidence = (
        estimate_confidence(
            backend_quality=backend_quality,
            estimated_noise=estimated_noise,
            estimated_drift=estimated_drift,
            confidence_error_std=(
                confidence_error_std
            ),
        )
    )

    return Scenario(
        scenario_id=scenario_id,
        true_value=true_value,
        predicted_value=predicted_value,
        confidence=confidence,

        true_noise=true_noise,
        true_drift=true_drift,
        signed_drift=signed_drift,

        noise=estimated_noise,
        drift=estimated_drift,

        delay_s=delay_s,

        backend_quality=backend_quality,
        state_velocity=state_velocity,
        value_at_generation=value_at_generation,
        staleness_error=staleness_error,

        absolute_error=absolute_error,

        bad_output=(
            absolute_error
            > error_threshold
        ),
    )


def make_random_scenario(
    scenario_id: int,
    true_value: float,
    error_threshold: float,
    correlated_confidence: bool = True,
    telemetry_estimation_std: float = 0.05,
    confidence_error_std: float = 0.05,
    max_state_change_rate_per_s: float = 0.02,
) -> Scenario:
    """
    Generate one randomized stress-test scenario.
    """

    true_noise = random.uniform(
        0.0,
        1.0,
    )

    true_drift = random.uniform(
        0.0,
        1.0,
    )

    delay_s = random.uniform(
        0.0,
        20.0,
    )

    signed_drift = (
        generate_signed_drift(
            true_drift
        )
    )

    backend_quality = random.uniform(
        0.30,
        1.0,
    )

    state_velocity = (
        generate_state_velocity(
            max_state_change_rate_per_s
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

    predicted_value = (
        value_at_generation
        + random.gauss(
            0.0,
            true_noise,
        )
        + signed_drift
    )

    absolute_error = abs(
        predicted_value
        - true_value
    )

    estimated_noise = (
        estimate_quality_indicator(
            true_noise,
            telemetry_estimation_std,
        )
    )

    estimated_drift = (
        estimate_quality_indicator(
            true_drift,
            telemetry_estimation_std,
        )
    )

    if correlated_confidence:

        confidence = (
            estimate_confidence(
                backend_quality=backend_quality,
                estimated_noise=estimated_noise,
                estimated_drift=estimated_drift,
                confidence_error_std=(
                    confidence_error_std
                ),
            )
        )

    else:

        confidence = random.uniform(
            0.0,
            1.0,
        )

    return Scenario(
        scenario_id=scenario_id,
        true_value=true_value,
        predicted_value=predicted_value,
        confidence=confidence,

        true_noise=true_noise,
        true_drift=true_drift,
        signed_drift=signed_drift,

        noise=estimated_noise,
        drift=estimated_drift,

        delay_s=delay_s,

        backend_quality=backend_quality,
        state_velocity=state_velocity,
        value_at_generation=value_at_generation,
        staleness_error=staleness_error,

        absolute_error=absolute_error,

        bad_output=(
            absolute_error
            > error_threshold
        ),
    )