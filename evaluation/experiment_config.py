from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ExperimentConfig:
    seed: int = 42

    true_value: float = 1.0
    error_threshold: float = 0.30

    theta_accept: float = 0.75
    theta_reject: float = 0.35

    alpha: float = 0.25
    beta: float = 0.25
    gamma: float = 0.25
    delta: float = 0.25

    tau_s: float = 5.0

    repetitions: int = 30

    noise_values: tuple[float, ...] = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5)
    drift_values: tuple[float, ...] = (0.0, 0.1, 0.2, 0.3, 0.4)
    delay_values_s: tuple[float, ...] = (0.1, 1.0, 3.0, 5.0, 10.0, 20.0)
    tau_values_s: tuple[float, ...] = (1.0, 5.0, 10.0, 30.0)

    random_scenarios: int = 10_000
    scalability_scenarios: tuple[int, ...] = (1_000, 10_000, 100_000)
    source_counts: tuple[int, ...] = (2, 5, 10, 25, 50, 100)