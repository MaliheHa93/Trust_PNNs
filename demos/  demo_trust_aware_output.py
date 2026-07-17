# demos/demo_trust_aware_output.py

from demos.common import build_default_orchestrator, make_chemical_task
from core.trust_manager import TrustManager, TrustConfig
from core.trust_models import TrustDecision


def main() -> None:
    orchestrator = build_default_orchestrator()

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

    task = make_chemical_task(
        task_id="trust-demo-chemical",
        input_level=1.4,
    )

    run_result = orchestrator.execute_task(task)

    if not run_result.success or run_result.invocation is None:
        print("phys-MCP execution failed:", run_result.failure_reason)
        return

    trust_result = trust_manager.evaluate(
        invocation=run_result.invocation,
        telemetry_after=run_result.telemetry_after,
    )

    print("Backend:", trust_result.evidence.backend_id)
    print("Value:", trust_result.evidence.value)
    print("Confidence:", trust_result.evidence.confidence)
    print("Noise:", trust_result.evidence.noise)
    print("Drift:", trust_result.evidence.drift)
    print("Freshness:", trust_result.evidence.freshness)
    print("Trust:", trust_result.evidence.trust)
    print("Decision:", trust_result.decision)
    print("Reason:", trust_result.reason)

    if trust_result.decision == TrustDecision.ACCEPT:
        print("Final output accepted at the edge.")
    elif trust_result.decision == TrustDecision.REJECT:
        print("Output rejected as unreliable.")
    else:
        print("Output should be forwarded to fog for fusion.")


if __name__ == "__main__":
    main()