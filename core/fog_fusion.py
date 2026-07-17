from __future__ import annotations

from core.trust_models import EvidenceRecord


class FogFusionEngine:
    def __init__(self, epsilon: float = 1e-6) -> None:
        self.epsilon = epsilon

    def are_compatible(self, records: list[EvidenceRecord]) -> bool:
        if not records:
            return False

        first_task_id = records[0].task_id
        first_modality = records[0].modality

        for record in records:
            if record.task_id != first_task_id:
                return False

            if record.modality != first_modality:
                return False

        return True

    def fuse_numeric(self, records: list[EvidenceRecord]) -> float:
        if not records:
            raise ValueError("Cannot fuse an empty list of evidence records.")

        if not self.are_compatible(records):
            raise ValueError(
                "Evidence records are not compatible. They must have the same task_id and modality."
            )

        weights: list[float] = []

        for record in records:
            weight = record.trust / ((record.noise ** 2) + self.epsilon)
            weights.append(weight)

        denominator = sum(weights)

        if denominator == 0:
            raise ValueError("Fusion denominator is zero.")

        numerator = 0.0

        for weight, record in zip(weights, records):
            numerator += weight * float(record.value)

        return numerator / denominator