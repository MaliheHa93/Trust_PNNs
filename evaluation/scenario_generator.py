from __future__ import annotations

import random
from dataclasses import dataclass


@dataclass
class Scenario:
    scenario_id: int
    true_value: float
    predicted_value: float
    confidence: float
    noise: float
    drift: float
    delay_s: float
    absolute_error: float
    bad_output: bool


def clip01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def make_controlled_scenario(
    scenario_id: int,
    true_value: float,
    noise: float,
    drift: float,
    delay_s: float,
    error_threshold: float,
) -> Scenario:
    random_noise = random.gauss(0.0, noise)
    predicted_value = true_value + random_noise + drift
    absolute_error = abs(predicted_value - true_value)

    confidence = clip01(1.0 - 0.5 * noise - 0.5 * drift)

    return Scenario(
        scenario_id=scenario_id,
        true_value=true_value,
        predicted_value=predicted_value,
        confidence=confidence,
        noise=clip01(noise),
        drift=clip01(drift),
        delay_s=delay_s,
        absolute_error=absolute_error,
        bad_output=absolute_error > error_threshold,
    )


def make_random_scenario(
    scenario_id: int,
    true_value: float,
    error_threshold: float,
    correlated_confidence: bool = True,
) -> Scenario:
    noise = random.uniform(0.0, 1.0)
    drift = random.uniform(0.0, 1.0)
    delay_s = random.uniform(0.0, 20.0)

    random_noise = random.gauss(0.0, noise)
    predicted_value = true_value + random_noise + drift
    absolute_error = abs(predicted_value - true_value)

    if correlated_confidence:
        confidence = 1.0 - 0.5 * noise - 0.5 * drift
        confidence += random.gauss(0.0, 0.05)
        confidence = clip01(confidence)
    else:
        confidence = random.uniform(0.0, 1.0)

    return Scenario(
        scenario_id=scenario_id,
        true_value=true_value,
        predicted_value=predicted_value,
        confidence=confidence,
        noise=clip01(noise),
        drift=clip01(drift),
        delay_s=delay_s,
        absolute_error=absolute_error,
        bad_output=absolute_error > error_threshold,
    )