from __future__ import annotations

from dataclasses import dataclass

from core.adaptive_trust import (
    AdaptiveTrustConfig,
    AdaptiveTrustTracker,
)

from core.orchestrator import (
    OrchestrationRunResult,
    PhysMCPOrchestrator,
)

from core.task_model import (
    TaskRequest,
)

from core.trust_manager import (
    TrustConfig,
    TrustManager,
)

from core.trust_models import (
    TrustResult,
)


@dataclass
class TrustAwareRunResult:
    """
    Complete result of:

        discovery/matching
        -> invocation
        -> trust evaluation
        -> historical trust update
        -> adaptive recovery
    """

    orchestration_result: (
        OrchestrationRunResult
    )

    trust_result: (
        TrustResult
        | None
    )

    historical_trust: (
        float
        | None
    )

    recalibration_requested: bool

    recalibration_triggered: bool

    recalibration_success: bool | None


class TrustAwarePhysMCPOrchestrator:

    def __init__(
        self,
        orchestrator: (
            PhysMCPOrchestrator
            | None
        ) = None,

        trust_manager: (
            TrustManager
            | None
        ) = None,

        adaptive_tracker: (
            AdaptiveTrustTracker
            | None
        ) = None,
    ) -> None:

        self.orchestrator = (
            orchestrator
            or PhysMCPOrchestrator()
        )

        self.trust_manager = (
            trust_manager
            or TrustManager(
                TrustConfig()
            )
        )

        self.adaptive_tracker = (
            adaptive_tracker
            or AdaptiveTrustTracker(
                AdaptiveTrustConfig()
            )
        )


    def execute_task(
        self,
        task: TaskRequest,
    ) -> TrustAwareRunResult:
        """
        Execute a task and apply post-invocation trust management.
        """

        base_run = (
            self.orchestrator
            .execute_task(
                task
            )
        )

        # -------------------------------------------------------------
        # No successful physical invocation -> no trust evaluation.
        # -------------------------------------------------------------

        if (
            not base_run.success

            or base_run.invocation
            is None
        ):

            return TrustAwareRunResult(

                orchestration_result=(
                    base_run
                ),

                trust_result=None,

                historical_trust=None,

                recalibration_requested=False,

                recalibration_triggered=False,

                recalibration_success=None,
            )

        # -------------------------------------------------------------
        # Post-invocation trust evaluation.
        # -------------------------------------------------------------

        trust_result = (
            self.trust_manager
            .evaluate_invocation(

                invocation=(
                    base_run.invocation
                ),

                telemetry_after=(
                    base_run.telemetry_after
                ),
            )
        )

        backend_id = (
            trust_result
            .evidence
            .backend_id
        )

        current_trust = (
            trust_result
            .evidence
            .trust
        )

        # -------------------------------------------------------------
        # Update historical trust.
        # -------------------------------------------------------------

        historical_trust = (
            self.adaptive_tracker
            .update(

                backend_id=(
                    backend_id
                ),

                current_trust=(
                    current_trust
                ),
            )
        )

        # -------------------------------------------------------------
        # Adaptive recalibration decision.
        # -------------------------------------------------------------

        recalibration_requested = (
            self.adaptive_tracker
            .requires_recalibration(
                backend_id
            )
        )

        recalibration_triggered = (
            False
        )

        recalibration_success: (
            bool
            | None
        ) = None

        if recalibration_requested:

            # Avoid duplicate recalibration if the legacy phys-MCP
            # recovery path already recalibrated this invocation.
            already_recalibrated = any(

                "recalibration"
                in action.lower()

                for action
                in base_run
                .recovery_actions
            )

            if not already_recalibrated:

                recalibration_triggered = (
                    True
                )

                recalibration_success = (
                    self.orchestrator
                    .recalibrate_backend(
                        backend_id
                    )
                )

            else:

                recalibration_triggered = (
                    True
                )

                recalibration_success = (
                    True
                )

            if recalibration_success:

                self.adaptive_tracker \
                    .reset_after_recalibration(
                        backend_id
                    )

        return TrustAwareRunResult(

            orchestration_result=(
                base_run
            ),

            trust_result=(
                trust_result
            ),

            historical_trust=(
                historical_trust
            ),

            recalibration_requested=(
                recalibration_requested
            ),

            recalibration_triggered=(
                recalibration_triggered
            ),

            recalibration_success=(
                recalibration_success
            ),
        )