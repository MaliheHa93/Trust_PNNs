"""Demo 2: invocation, telemetry, lifecycle, and recovery."""

from __future__ import annotations

from demos.common import (
    build_default_orchestrator,
    make_chemical_task,
    print_header,
    print_run_summary,
)

from core.trust_manager import TrustConfig, TrustManager


def main() -> None:
    orchestrator = build_default_orchestrator()
    task = make_chemical_task(task_id="chemical-lifecycle-demo", input_level=1.8)

    trust_manager = TrustManager(
        TrustConfig(
            alpha=0.25,
            beta=0.25,
            gamma=0.25,
            delta=0.25,
            tau_s=5.0,
            theta_accept=0.75,
            theta_reject=0.35,
        )
    )

    print_header("Initial chemical backend telemetry")
    print(orchestrator.registry.get_adapter("chemical-backend").collect_telemetry())

    print_header("Repeated invocations until lifecycle recovery is triggered")

    for cycle in range(1, 13):
        run_result = orchestrator.execute_task(task)

        print(f"\nCycle {cycle}")
        print_run_summary(run_result)
        print("-" * 80)

        if run_result.success and run_result.invocation is not None:
            trust_result = trust_manager.evaluate_invocation(
                invocation=run_result.invocation,
                telemetry_after=run_result.telemetry_after,
            )

            evidence = trust_result.evidence

            print("\n========== TRUST-AWARE OUTPUT MANAGEMENT ==========")
            print("Backend:", evidence.backend_id)
            print("Task:", evidence.task_id)
            print("Value:", evidence.value)
            print("Confidence:", evidence.confidence)
            print("Noise:", evidence.noise)
            print("Drift:", evidence.drift)
            print("Freshness:", evidence.freshness)
            print("Trust:", evidence.trust)
            print("Decision:", trust_result.decision.value)
            print("Reason:", trust_result.reason)
        else:
            print("No successful invocation, so trust score cannot be computed.")

        if run_result.recovery_actions:
            print("Lifecycle recovery has been demonstrated. Stopping the loop.")
            break


if __name__ == "__main__":
    main()