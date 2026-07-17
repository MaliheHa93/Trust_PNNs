from types import SimpleNamespace

from core.trust_manager import TrustConfig, TrustManager
from core.trust_models import TrustDecision


def test_high_quality_output_is_accepted():
    trust_manager = TrustManager(
        TrustConfig(
            tau_s=5.0,
            theta_accept=0.75,
            theta_reject=0.35,
        )
    )

    invocation = SimpleNamespace(
        backend_id="test-backend",
        task_id="test-task",
        output_payload={"value": 1.0, "modality": "test"},
        confidence=0.95,
        execution_latency_ms=100.0,
        backend_state="ready",
        notes=None,
    )

    telemetry = {
        "noise_score": 0.05,
        "drift_score": 0.05,
        "age_of_information_ms": 100.0,
        "output_modality": "test",
    }

    result = trust_manager.evaluate_invocation(invocation, telemetry)

    assert result.decision == TrustDecision.ACCEPT


def test_bad_output_is_rejected():
    trust_manager = TrustManager(
        TrustConfig(
            tau_s=5.0,
            theta_accept=0.75,
            theta_reject=0.35,
        )
    )

    invocation = SimpleNamespace(
        backend_id="test-backend",
        task_id="test-task",
        output_payload={"value": 1.0, "modality": "test"},
        confidence=0.10,
        execution_latency_ms=20000.0,
        backend_state="degraded",
        notes=None,
    )

    telemetry = {
        "noise_score": 0.95,
        "drift_score": 0.90,
        "age_of_information_ms": 20000.0,
        "output_modality": "test",
    }

    result = trust_manager.evaluate_invocation(invocation, telemetry)

    assert result.decision == TrustDecision.REJECT