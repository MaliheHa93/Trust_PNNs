# core/trust_models.py

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class TrustDecision(str, Enum):
    ACCEPT = "accept"
    REJECT = "reject"
    FORWARD_TO_FOG = "forward_to_fog"


@dataclass
class EvidenceRecord:
    backend_id: str
    task_id: str
    value: Any
    confidence: float
    noise: float
    drift: float
    timestamp: float
    modality: str
    provenance: dict[str, Any] = field(default_factory=dict)
    freshness: float = 0.0
    trust: float = 0.0


@dataclass
class TrustResult:
    evidence: EvidenceRecord
    decision: TrustDecision
    reason: str