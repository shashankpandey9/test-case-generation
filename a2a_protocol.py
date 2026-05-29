"""
A2A (Agent-to-Agent) Protocol
Lightweight message-passing protocol for inter-agent communication in the
test-case generation pipeline.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional


@dataclass
class A2AMessage:
    """Envelope for every inter-agent message."""
    sender: str
    receiver: str
    message_type: str          # data_context | requirements | test_cases | review_result | output
    payload: dict
    trace_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> dict:
        return {
            "sender": self.sender,
            "receiver": self.receiver,
            "message_type": self.message_type,
            "payload": self.payload,
            "trace_id": self.trace_id,
            "timestamp": self.timestamp,
        }


class A2AProtocol:
    """
    In-memory message bus shared across all agents within a single pipeline run.
    Each run should instantiate a fresh A2AProtocol with a unique trace_id.
    """

    def __init__(self, trace_id: Optional[str] = None):
        self.trace_id: str = trace_id or str(uuid.uuid4())
        # inbox: agent_name -> list of messages received
        self._inbox: Dict[str, List[A2AMessage]] = {}
        # full ordered trace of every message sent
        self._trace: List[A2AMessage] = []

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def send(self, message: A2AMessage) -> None:
        """Route a message to the receiver's inbox and append to trace."""
        message.trace_id = self.trace_id          # stamp with run trace_id
        self._inbox.setdefault(message.receiver, []).append(message)
        self._trace.append(message)
        print(
            f"[A2A] {message.sender} → {message.receiver} "
            f"[{message.message_type}] @ {message.timestamp}"
        )

    def receive(self, agent_name: str) -> Optional[A2AMessage]:
        """Pop the latest message from an agent's inbox (LIFO)."""
        inbox = self._inbox.get(agent_name, [])
        return inbox.pop() if inbox else None

    def get_trace(self, trace_id: Optional[str] = None) -> List[dict]:
        """Return the full ordered message trace as a list of dicts."""
        return [m.to_dict() for m in self._trace]

    def reset(self) -> None:
        """Clear inbox and trace (useful for retries)."""
        self._inbox.clear()
        self._trace.clear()
