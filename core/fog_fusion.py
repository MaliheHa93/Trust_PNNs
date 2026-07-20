from __future__ import annotations

import math
from dataclasses import dataclass
from statistics import median

from core.trust_models import EvidenceRecord


@dataclass
class FogFusionConfig:
    """
    Configuration for fog-level compatibility checking,
    disagreement assessment, and numerical fusion.
    """

    # Prevent unrealistically extreme inverse-uncertainty weights.
    sigma_floor: float = 0.02

    # Numerical stability constant.
    epsilon: float = 1e-6

    # Maximum timestamp separation between evidence records.
    max_timestamp_difference_s: float = 5.0

    # Maximum normalized disagreement before fusion is blocked.
    max_normalized_disagreement: float = 0.75

    # Fog-specific reliability weights.
    #
    # R_i =
    #   confidence_weight * c_i
    #   + drift_weight * (1 - d_i)
    #   + freshness_weight * f_i
    #
    # Noise is excluded because it is incorporated separately
    # through inverse-uncertainty weighting.
    confidence_weight: float = 1.0 / 3.0
    drift_weight: float = 1.0 / 3.0
    freshness_weight: float = 1.0 / 3.0

    def __post_init__(self) -> None:
        """
        Validate fusion configuration.
        """

        if self.sigma_floor <= 0.0:
            raise ValueError(
                "sigma_floor must be greater than zero."
            )

        if self.epsilon <= 0.0:
            raise ValueError(
                "epsilon must be greater than zero."
            )

        if self.max_timestamp_difference_s < 0.0:
            raise ValueError(
                "max_timestamp_difference_s "
                "cannot be negative."
            )

        if self.max_normalized_disagreement < 0.0:
            raise ValueError(
                "max_normalized_disagreement "
                "cannot be negative."
            )

        weights = [
            self.confidence_weight,
            self.drift_weight,
            self.freshness_weight,
        ]

        if any(
            weight < 0.0
            for weight in weights
        ):
            raise ValueError(
                "Fog reliability weights "
                "must be non-negative."
            )

        if not math.isclose(
            sum(weights),
            1.0,
            rel_tol=1e-9,
            abs_tol=1e-9,
        ):
            raise ValueError(
                "Fog reliability weights must sum to 1."
            )


@dataclass
class FusionAssessment:
    """
    Result of compatibility and disagreement assessment.
    """

    compatible: bool

    compatibility_issues: list[str]

    disagreement: float

    disagreement_acceptable: bool

    can_fuse: bool


class FogFusionEngine:
    """
    Fog-level compatibility checking, disagreement assessment,
    and reliability-aware numerical fusion.
    """

    def __init__(
        self,
        config: FogFusionConfig | None = None,
    ) -> None:

        self.config = (
            config
            or FogFusionConfig()
        )

    # =============================================================
    # Utility
    # =============================================================

    @staticmethod
    def _clip01(
        value: float,
    ) -> float:
        """
        Clip normalized values to [0, 1].
        """

        return max(
            0.0,
            min(
                1.0,
                float(value),
            ),
        )

    # =============================================================
    # Compatibility checking
    # =============================================================

    def compatibility_issues(
        self,
        records: list[
            EvidenceRecord
        ],
    ) -> list[str]:
        """
        Determine whether evidence records are compatible for
        fog-level comparison and fusion.

        Compatibility considers:
            - task identifier,
            - output modality,
            - timestamp window,
            - decision context when used,
            - observation-window identifier when used.
        """

        if not records:
            return [
                "No evidence records provided."
            ]

        issues: list[str] = []

        first = records[0]

        # ---------------------------------------------------------
        # Task compatibility
        # ---------------------------------------------------------

        if any(
            record.task_id
            != first.task_id

            for record
            in records
        ):
            issues.append(
                "Evidence records have different task IDs."
            )

        # ---------------------------------------------------------
        # Modality compatibility
        # ---------------------------------------------------------

        if any(
            record.modality
            != first.modality

            for record
            in records
        ):
            issues.append(
                "Evidence records have different modalities."
            )

        # ---------------------------------------------------------
        # Observation-time compatibility
        # ---------------------------------------------------------

        timestamps = [
            float(
                record.timestamp
            )

            for record
            in records
        ]

        timestamp_range = (
            max(timestamps)
            - min(timestamps)
        )

        if (
            timestamp_range
            > self.config
            .max_timestamp_difference_s
        ):
            issues.append(
                "Evidence records come from "
                "incompatible observation times."
            )

        # ---------------------------------------------------------
        # Decision-context compatibility
        #
        # If context information is present for any record,
        # require it to be available and identical for all records.
        # ---------------------------------------------------------

        contexts = [
            record
            .provenance
            .get(
                "decision_context"
            )

            for record
            in records
        ]

        available_contexts = [
            context

            for context
            in contexts

            if context
            is not None
        ]

        if available_contexts:

            if (
                len(
                    available_contexts
                )
                != len(
                    records
                )
            ):
                issues.append(
                    "Decision-context provenance "
                    "is missing from one or more "
                    "evidence records."
                )

            elif (
                len(
                    set(
                        available_contexts
                    )
                )
                > 1
            ):
                issues.append(
                    "Evidence records belong to "
                    "different decision contexts."
                )

        # ---------------------------------------------------------
        # Observation-window compatibility
        #
        # If an observation-window identifier is present for any
        # record, require it to be present and identical for all.
        # ---------------------------------------------------------

        windows = [
            record
            .provenance
            .get(
                "observation_window_id"
            )

            for record
            in records
        ]

        available_windows = [
            window

            for window
            in windows

            if window
            is not None
        ]

        if available_windows:

            if (
                len(
                    available_windows
                )
                != len(
                    records
                )
            ):
                issues.append(
                    "Observation-window provenance "
                    "is missing from one or more "
                    "evidence records."
                )

            elif (
                len(
                    set(
                        available_windows
                    )
                )
                > 1
            ):
                issues.append(
                    "Evidence records belong to "
                    "different observation windows."
                )

        return issues

    def are_compatible(
        self,
        records: list[
            EvidenceRecord
        ],
    ) -> bool:
        """
        Return True when no compatibility issue is detected.
        """

        return not (
            self.compatibility_issues(
                records
            )
        )

    # =============================================================
    # Disagreement assessment
    # =============================================================

    def compute_disagreement(
        self,
        records: list[
            EvidenceRecord
        ],
    ) -> float:
        """
        Compute robust normalized disagreement.

                    max_i |v_i - median(v)|
        Delta_fog = -----------------------
                       max(|median(v)|, 1)
        """

        if not records:
            raise ValueError(
                "Cannot compute disagreement "
                "for an empty record list."
            )

        try:

            values = [
                float(
                    record.value
                )

                for record
                in records
            ]

        except (
            TypeError,
            ValueError,
        ) as exc:

            raise ValueError(
                "Fog disagreement assessment "
                "requires numerical evidence values."
            ) from exc

        center = float(
            median(
                values
            )
        )

        maximum_deviation = max(
            abs(
                value
                - center
            )

            for value
            in values
        )

        scale = max(
            abs(
                center
            ),
            1.0,
        )

        return (
            maximum_deviation
            / scale
        )

    def assess(
        self,
        records: list[
            EvidenceRecord
        ],
    ) -> FusionAssessment:
        """
        Perform compatibility and disagreement assessment.
        """

        issues = (
            self.compatibility_issues(
                records
            )
        )

        compatible = (
            len(
                issues
            )
            == 0
        )

        if not records:

            return FusionAssessment(
                compatible=False,
                compatibility_issues=issues,
                disagreement=float(
                    "inf"
                ),
                disagreement_acceptable=False,
                can_fuse=False,
            )

        disagreement = (
            self.compute_disagreement(
                records
            )
        )

        disagreement_acceptable = (
            disagreement
            <= self.config
            .max_normalized_disagreement
        )

        return FusionAssessment(
            compatible=compatible,
            compatibility_issues=issues,
            disagreement=disagreement,
            disagreement_acceptable=(
                disagreement_acceptable
            ),
            can_fuse=(
                compatible
                and disagreement_acceptable
            ),
        )

    # =============================================================
    # Uncertainty handling
    # =============================================================

    def _effective_noise(
        self,
        record: EvidenceRecord,
    ) -> float:
        """
        Apply the uncertainty floor.
        """

        noise = self._clip01(
            record.noise
        )

        return max(
            noise,
            self.config
            .sigma_floor,
        )

    # =============================================================
    # Fog-level reliability
    # =============================================================

    def compute_fusion_reliability(
        self,
        record: EvidenceRecord,
    ) -> float:
        """
        Compute fog-level reliability:

        R_i =
            alpha_f * c_i
            + gamma_f * (1 - d_i)
            + delta_f * f_i

        Noise is excluded because it is incorporated separately
        through inverse-uncertainty weighting.
        """

        cfg = self.config

        confidence = self._clip01(
            record.confidence
        )

        drift = self._clip01(
            record.drift
        )

        freshness = self._clip01(
            record.freshness
        )

        reliability = (

            cfg.confidence_weight
            * confidence

            + cfg.drift_weight
            * (
                1.0
                - drift
            )

            + cfg.freshness_weight
            * freshness
        )

        return self._clip01(
            reliability
        )

    # =============================================================
    # Proposed numerical fusion
    # =============================================================

    def fuse_numeric(
        self,
        records: list[
            EvidenceRecord
        ],
        enforce_assessment: bool = True,
    ) -> float:
        """
        Proposed fusion formulation:

                    R_i
        w_i = -------------------
              sigma_i^2 + epsilon

        where sigma_i is bounded below by sigma_floor.

        When enforce_assessment=True, compatibility and
        disagreement gates are applied before fusion.
        """

        if not records:
            raise ValueError(
                "Cannot fuse an empty "
                "list of evidence records."
            )

        if enforce_assessment:

            assessment = (
                self.assess(
                    records
                )
            )

            if not (
                assessment.compatible
            ):
                raise ValueError(
                    "Evidence is incompatible: "
                    + "; ".join(
                        assessment
                        .compatibility_issues
                    )
                )

            if not (
                assessment
                .disagreement_acceptable
            ):
                raise ValueError(
                    "Evidence disagreement exceeds "
                    "the configured fusion threshold."
                )

        weights: list[
            float
        ] = []

        for record in records:

            sigma = (
                self._effective_noise(
                    record
                )
            )

            reliability = (
                self.compute_fusion_reliability(
                    record
                )
            )

            weight = (
                reliability
                / (
                    sigma ** 2
                    + self.config
                    .epsilon
                )
            )

            weights.append(
                weight
            )

        denominator = sum(
            weights
        )

        if denominator <= 0.0:
            raise ValueError(
                "Fusion denominator is zero."
            )

        numerator = sum(

            weight
            * float(
                record.value
            )

            for (
                weight,
                record,
            )

            in zip(
                weights,
                records,
            )
        )

        return (
            numerator
            / denominator
        )

    # =============================================================
    # Original fusion formulation retained for comparison
    # =============================================================

    def fuse_numeric_original_trust(
        self,
        records: list[
            EvidenceRecord
        ],
        enforce_assessment: bool = True,
    ) -> float:
        """
        Original comparison formulation:

                    T_i
        w_i = -------------------
              sigma_i^2 + epsilon

        This is retained as an experimental baseline because
        T_i already incorporates noise, potentially counting
        uncertainty twice.
        """

        if not records:
            raise ValueError(
                "Cannot fuse an empty "
                "list of evidence records."
            )

        if enforce_assessment:

            assessment = (
                self.assess(
                    records
                )
            )

            if not (
                assessment.can_fuse
            ):
                raise ValueError(
                    "Evidence cannot safely be fused."
                )

        weights: list[
            float
        ] = []

        for record in records:

            sigma = (
                self._effective_noise(
                    record
                )
            )

            trust = self._clip01(
                record.trust
            )

            weight = (
                trust
                / (
                    sigma ** 2
                    + self.config
                    .epsilon
                )
            )

            weights.append(
                weight
            )

        denominator = sum(
            weights
        )

        if denominator <= 0.0:
            raise ValueError(
                "Fusion denominator is zero."
            )

        numerator = sum(

            weight
            * float(
                record.value
            )

            for (
                weight,
                record,
            )

            in zip(
                weights,
                records,
            )
        )

        return (
            numerator
            / denominator
        )