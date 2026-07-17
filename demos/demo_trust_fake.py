from __future__ import annotations

from types import SimpleNamespace

from core.trust_manager import TrustConfig, TrustManager


def main() -> None:
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

    fake_invocation = SimpleNamespace(
        backend_id="chemical-backend",
        task_id="task-001",
        output_payload={
            "value": 1.25,
            "modality": "chemical_concentration",
        },
        confidence=0.90,
        execution_latency_ms=1200.0,
        backend_state="ready",
        notes="Fake invocation for testing trust manager.",
    )

    fake_telemetry_after = {
        "noise_score": 0.10,
        "drift_score": 0.05,
        "age_of_information_ms": 1200.0,
        "output_modality": "chemical_concentration",
        "health_status": "healthy",
    }
    #fake_telemetry_after = {
     
     # "noise_score": 0.9,
      #  "drift_score": 0.8,
       # "age_of_information_ms": 2000.0,
        #"output_modality": "chemical_concentration",
        #"health_status": "degraded",
    #}

    trust_result = trust_manager.evaluate_invocation(
        invocation=fake_invocation,
        telemetry_after=fake_telemetry_after,
    )

    evidence = trust_result.evidence

    print("========== TRUST RESULT ==========")
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


if __name__ == "__main__":
    main()