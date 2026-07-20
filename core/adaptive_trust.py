from __future__ import annotations

from dataclasses import dataclass, field


# ---------------------------------------------------------------------
# Adaptive trust configuration
# ---------------------------------------------------------------------

@dataclass
class AdaptiveTrustConfig:
    """
    Configuration for historical trust tracking and recalibration.

    lambda_history:
        Weight assigned to previous historical trust in the
        exponential moving average.

        historical_trust(t) =
            lambda * historical_trust(t-1)
            + (1-lambda) * current_trust(t)

    low_trust_threshold:
        Instantaneous trust below this threshold is considered
        a low-trust observation.

    historical_trust_threshold:
        Historical trust below this threshold indicates persistent
        long-term degradation.

    low_trust_streak_limit:
        Number of consecutive low-trust observations required to
        trigger recalibration.
    """

    lambda_history: float = 0.8

    low_trust_threshold: float = 0.60

    historical_trust_threshold: float = 0.60

    low_trust_streak_limit: int = 3


# ---------------------------------------------------------------------
# Adaptive trust state
# ---------------------------------------------------------------------

@dataclass
class AdaptiveTrustState:
    """
    Maintains historical trust and low-trust streaks for each backend.
    """

    historical_trust: dict[
        str,
        float,
    ] = field(
        default_factory=dict
    )

    low_trust_streak: dict[
        str,
        int,
    ] = field(
        default_factory=dict
    )


# ---------------------------------------------------------------------
# Adaptive trust tracker
# ---------------------------------------------------------------------

class AdaptiveTrustTracker:
    """
    Tracks long-term reliability of PNN backends.

    The tracker maintains an exponential moving average of trust and
    detects persistent degradation.

    Recalibration can be requested when either:

    1. Historical trust falls below the configured reliability
       threshold.

    OR

    2. A backend generates several consecutive low-trust outputs.
    """

    def __init__(
        self,
        config: AdaptiveTrustConfig | None = None,
    ) -> None:

        self.config = (
            config
            or AdaptiveTrustConfig()
        )

        self.state = (
            AdaptiveTrustState()
        )

        self._validate_config()

    # -----------------------------------------------------------------
    # Configuration validation
    # -----------------------------------------------------------------

    def _validate_config(
        self,
    ) -> None:
        """
        Validate adaptive trust configuration.
        """

        cfg = self.config

        if not (
            0.0
            <= cfg.lambda_history
            <= 1.0
        ):
            raise ValueError(
                "lambda_history must be "
                "between 0 and 1."
            )

        if not (
            0.0
            <= cfg.low_trust_threshold
            <= 1.0
        ):
            raise ValueError(
                "low_trust_threshold must be "
                "between 0 and 1."
            )

        if not (
            0.0
            <= cfg.historical_trust_threshold
            <= 1.0
        ):
            raise ValueError(
                "historical_trust_threshold "
                "must be between 0 and 1."
            )

        if (
            cfg.low_trust_streak_limit
            < 1
        ):
            raise ValueError(
                "low_trust_streak_limit must "
                "be at least 1."
            )

    # -----------------------------------------------------------------
    # Historical trust update
    # -----------------------------------------------------------------

    def update(
        self,
        backend_id: str,
        current_trust: float,
    ) -> float:
        """
        Update historical trust for one backend.

        Historical trust is calculated using an exponential
        moving average:

            H_t =
                lambda * H_(t-1)
                + (1-lambda) * T_t

        For the first observation, current trust is used as the
        initial historical trust value.
        """

        current_trust = max(
            0.0,
            min(
                1.0,
                float(
                    current_trust
                ),
            ),
        )

        cfg = self.config

        # -------------------------------------------------------------
        # Obtain previous historical trust.
        #
        # For the first observation, initialize historical trust with
        # current trust.
        # -------------------------------------------------------------

        previous_historical_trust = (
            self.state.historical_trust.get(
                backend_id,
                current_trust,
            )
        )

        # -------------------------------------------------------------
        # Exponential moving average
        # -------------------------------------------------------------

        new_historical_trust = (
            cfg.lambda_history
            * previous_historical_trust
            + (
                1.0
                - cfg.lambda_history
            )
            * current_trust
        )

        self.state.historical_trust[
            backend_id
        ] = new_historical_trust

        # -------------------------------------------------------------
        # Update consecutive low-trust counter
        # -------------------------------------------------------------

        if (
            current_trust
            < cfg.low_trust_threshold
        ):

            previous_streak = (
                self.state.low_trust_streak.get(
                    backend_id,
                    0,
                )
            )

            self.state.low_trust_streak[
                backend_id
            ] = (
                previous_streak
                + 1
            )

        else:

            # Reset streak when current trust returns above threshold.
            self.state.low_trust_streak[
                backend_id
            ] = 0

        return (
            new_historical_trust
        )

    # -----------------------------------------------------------------
    # Recalibration decision
    # -----------------------------------------------------------------

    def requires_recalibration(
        self,
        backend_id: str,
    ) -> bool:
        """
        Determine whether a backend should be recalibrated.

        Recalibration is requested when either:

        1. Historical trust falls below the configured historical
           trust threshold.

        OR

        2. The backend generates the configured number of consecutive
           low-trust outputs.

        This allows the adaptive mechanism to react to both:

        - persistent long-term degradation, and
        - repeated short-term reliability failures.
        """

        cfg = self.config

        # -------------------------------------------------------------
        # Consecutive low-trust observations
        # -------------------------------------------------------------

        low_trust_streak = (
            self.state.low_trust_streak.get(
                backend_id,
                0,
            )
        )

        repeated_low_trust = (
            low_trust_streak
            >= cfg.low_trust_streak_limit
        )

        # -------------------------------------------------------------
        # Historical degradation
        # -------------------------------------------------------------

        historical_trust = (
            self.state.historical_trust.get(
                backend_id
            )
        )

        historical_degradation = (
            historical_trust
            is not None
            and historical_trust
            < cfg.historical_trust_threshold
        )

        # -------------------------------------------------------------
        # Either condition can trigger recalibration
        # -------------------------------------------------------------

        return (
            repeated_low_trust
            or historical_degradation
        )

    # -----------------------------------------------------------------
    # Historical trust lookup
    # -----------------------------------------------------------------

    def get_historical_trust(
        self,
        backend_id: str,
    ) -> float | None:
        """
        Return the historical trust value for a backend.

        Returns None if no observations have been recorded.
        """

        return (
            self.state.historical_trust.get(
                backend_id
            )
        )

    # -----------------------------------------------------------------
    # Low-trust streak lookup
    # -----------------------------------------------------------------

    def get_low_trust_streak(
        self,
        backend_id: str,
    ) -> int:
        """
        Return the current consecutive low-trust count.
        """

        return (
            self.state.low_trust_streak.get(
                backend_id,
                0,
            )
        )

    # -----------------------------------------------------------------
    # Reset after recalibration
    # -----------------------------------------------------------------

    def reset_after_recalibration(
        self,
        backend_id: str,
    ) -> None:
        """
        Reset the consecutive low-trust streak after recalibration.

        Historical trust is intentionally preserved so that the
        substrate does not immediately regain full long-term trust.
        It must recover gradually through subsequent reliable
        observations.
        """

        self.state.low_trust_streak[
            backend_id
        ] = 0

    # -----------------------------------------------------------------
    # Complete state reset
    # -----------------------------------------------------------------

    def reset_backend(
        self,
        backend_id: str,
    ) -> None:
        """
        Completely remove adaptive trust history for a backend.

        This should normally be used only when the backend is removed,
        replaced, or intentionally treated as a new substrate.
        """

        self.state.historical_trust.pop(
            backend_id,
            None,
        )

        self.state.low_trust_streak.pop(
            backend_id,
            None,
        )