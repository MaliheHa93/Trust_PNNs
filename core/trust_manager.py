from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any

from core.trust_models import EvidenceRecord, TrustDecision, TrustResult


@dataclass
class TrustConfig:
    alpha: float = 0.25
    beta: float = 0.25
    gamma: float = 0.25
    delta: float = 0.25

    tau_s: float = 5.0

    theta_accept: float = 0.75
    theta_reject: float = 0.35

    default_confidence: float = 0.5
    default_noise: float = 0.3
    default_drift: float = 0.0
    default_modality: str = "unknown"

    def __post_init__(self) -> None:
        weight_sum = self.alpha + self.beta + self.gamma + self.delta

        if abs(weight_sum - 1.0) > 1e-6:
            raise ValueError(
                "alpha + beta + gamma + delta must be equal to 1."
            )

        if self.tau_s <= 0:
            raise ValueError("tau_s must be larger than 0.")

        if self.theta_reject >= self.theta_accept:
            raise ValueError(
                "theta_reject must be smaller than theta_accept."
            )


class TrustManager:
    def __init__(self, config: TrustConfig | None = None) -> None:
        self.config = config or TrustConfig()

    @staticmethod
    def _clip01(value: float) -> float:
        return max(0.0, min(1.0, float(value)))

    @staticmethod
    def _get(obj: Any, key: str, default: Any = None) -> Any:
        if isinstance(obj, dict):
            return obj.get(key, default)

        return getattr(obj, key, default)

    @staticmethod
    def _get_first_number(data: dict[str, Any], keys: list[str], default: float) -> float:
        for key in keys:
            value = data.get(key)

            if isinstance(value, (int, float)):
                return float(value)

        return default

    def compute_freshness(self, age_s: float) -> float:
        age_s = max(0.0, float(age_s))
        return math.exp(-age_s / self.config.tau_s)

    def compute_trust(
        self,
        confidence: float,
        noise: float,
        drift: float,
        freshness: float,
    ) -> float:
        cfg = self.config

        confidence = self._clip01(confidence)
        noise = self._clip01(noise)
        drift = self._clip01(drift)
        freshness = self._clip01(freshness)

        trust = (
            cfg.alpha * confidence
            + cfg.beta * (1.0 - noise)
            + cfg.gamma * (1.0 - drift)
            + cfg.delta * freshness
        )

        return self._clip01(trust)

    def _extract_value(self, output_payload: Any) -> Any:
        if not isinstance(output_payload, dict):
            return output_payload

        preferred_keys = [
            "value",
            "result",
            "prediction",
            "response",
            "response_fingerprint",
        ]

        for key in preferred_keys:
            if key in output_payload:
                return output_payload[key]

        return output_payload

    def build_evidence_from_invocation(
        self,
        invocation: Any,
        telemetry_after: dict[str, Any],
    ) -> EvidenceRecord:
        now = time.time()

        output_payload = self._get(invocation, "output_payload", {})
        backend_id = self._get(invocation, "backend_id", "unknown_backend")
        task_id = self._get(invocation, "task_id", "unknown_task")
        backend_state = self._get(invocation, "backend_state", "unknown")
        notes = self._get(invocation, "notes", None)

        confidence = self._get(invocation, "confidence", None)

        if confidence is None:
            confidence = self._get_first_number(
                telemetry_after,
                keys=["confidence", "confidence_score"],
                default=self.config.default_confidence,
            )

        confidence = self._clip01(confidence)

        noise = self._get_first_number(
            telemetry_after,
            keys=["noise_score", "uncertainty_score", "output_noise", "noise"],
            default=self.config.default_noise,
        )

        drift = self._get_first_number(
            telemetry_after,
            keys=["drift_score", "drift", "substrate_drift"],
            default=self.config.default_drift,
        )

        noise = self._clip01(noise)
        drift = self._clip01(drift)

        generated_at_s = telemetry_after.get("generated_at_s")

        if isinstance(generated_at_s, (int, float)):
            timestamp = float(generated_at_s)
            age_s = max(0.0, now - timestamp)
        else:
            age_ms = telemetry_after.get("age_of_information_ms")

            if isinstance(age_ms, (int, float)):
                age_s = max(0.0, float(age_ms) / 1000.0)
            else:
                execution_latency_ms = self._get(
                    invocation,
                    "execution_latency_ms",
                    0.0,
                )
                age_s = max(0.0, float(execution_latency_ms) / 1000.0)

            timestamp = now - age_s

        freshness = self.compute_freshness(age_s)

        trust = self.compute_trust(
            confidence=confidence,
            noise=noise,
            drift=drift,
            freshness=freshness,
        )

        modality = (
            telemetry_after.get("output_modality")
            or telemetry_after.get("modality")
            or (
                output_payload.get("modality")
                if isinstance(output_payload, dict)
                else None
            )
            or self.config.default_modality
        )

        provenance = {
            "backend_id": backend_id,
            "backend_state": backend_state,
            "notes": notes,
            "telemetry_after": telemetry_after,
            "raw_output_payload": output_payload,
        }

        return EvidenceRecord(
            backend_id=str(backend_id),
            task_id=str(task_id),
            value=self._extract_value(output_payload),
            confidence=confidence,
            noise=noise,
            drift=drift,
            timestamp=timestamp,
            modality=str(modality),
            provenance=provenance,
            freshness=freshness,
            trust=trust,
        )

    def decide(self, evidence: EvidenceRecord) -> TrustResult:
        if evidence.trust >= self.config.theta_accept:
            return TrustResult(
                evidence=evidence,
                decision=TrustDecision.ACCEPT,
                reason="Trust score is high enough, so the output is accepted at the edge.",
            )

        if evidence.trust < self.config.theta_reject:
            return TrustResult(
                evidence=evidence,
                decision=TrustDecision.REJECT,
                reason="Trust score is too low, so the output is rejected.",
            )

        return TrustResult(
            evidence=evidence,
            decision=TrustDecision.FORWARD_TO_FOG,
            reason="Trust score is uncertain, so the output should be forwarded to the fog layer.",
        )

    def evaluate_invocation(
        self,
        invocation: Any,
        telemetry_after: dict[str, Any],
    ) -> TrustResult:
        evidence = self.build_evidence_from_invocation(
            invocation=invocation,
            telemetry_after=telemetry_after,
        )

        return self.decide(evidence)