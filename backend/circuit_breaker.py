"""
Circuit Breaker + Quarantine System
If final risk score >= 70, quarantine the agent:
  - Block outbound messages
  - Set to read-only
  - Broadcast quarantine signal to other agents
"""

import threading
import time
from enum import Enum
from typing import Dict, List, Optional, Callable


class AgentStatus(str, Enum):
    ACTIVE = "active"
    SUSPICIOUS = "suspicious"
    QUARANTINED = "quarantined"


RISK_THRESHOLDS = {
    "suspicious": 40,
    "quarantine": 70,
}


class CircuitBreaker:
    def __init__(self):
        self._agent_status: Dict[str, AgentStatus] = {}
        self._quarantine_log: List[dict] = []
        self._lock = threading.Lock()
        self._on_quarantine_callbacks: List[Callable] = []

    def register_agent(self, agent_id: str):
        with self._lock:
            self._agent_status[agent_id] = AgentStatus.ACTIVE

    def evaluate(self, agent_id: str, risk_score: float) -> dict:
        """
        Evaluate risk score and apply circuit breaker logic.
        Returns action taken.
        """
        with self._lock:
            prev_status = self._agent_status.get(agent_id, AgentStatus.ACTIVE)

            if risk_score >= RISK_THRESHOLDS["quarantine"]:
                new_status = AgentStatus.QUARANTINED
            elif risk_score >= RISK_THRESHOLDS["suspicious"]:
                new_status = AgentStatus.SUSPICIOUS
            else:
                new_status = AgentStatus.ACTIVE

            self._agent_status[agent_id] = new_status

            result = {
                "agent_id": agent_id,
                "prev_status": prev_status.value,
                "new_status": new_status.value,
                "risk_score": risk_score,
                "action_taken": "none",
                "timestamp": time.time(),
            }

            if new_status == AgentStatus.QUARANTINED and prev_status != AgentStatus.QUARANTINED:
                result["action_taken"] = "quarantine"
                result["details"] = {
                    "outbound_blocked": True,
                    "read_only": True,
                    "quarantine_signal_broadcast": True,
                }
                self._quarantine_log.append(result)
                # Fire callbacks
                for cb in self._on_quarantine_callbacks:
                    try:
                        cb(agent_id, result)
                    except Exception:
                        pass

            elif new_status == AgentStatus.SUSPICIOUS and prev_status == AgentStatus.ACTIVE:
                result["action_taken"] = "flag_suspicious"

            return result

    def is_quarantined(self, agent_id: str) -> bool:
        return self._agent_status.get(agent_id) == AgentStatus.QUARANTINED

    def get_status(self, agent_id: str) -> str:
        return self._agent_status.get(agent_id, AgentStatus.ACTIVE).value

    def get_all_statuses(self) -> Dict[str, str]:
        with self._lock:
            return {k: v.value for k, v in self._agent_status.items()}

    def release_quarantine(self, agent_id: str):
        with self._lock:
            self._agent_status[agent_id] = AgentStatus.ACTIVE

    def on_quarantine(self, callback: Callable):
        self._on_quarantine_callbacks.append(callback)

    def can_send_message(self, agent_id: str) -> bool:
        return not self.is_quarantined(agent_id)


# Global singleton
circuit_breaker = CircuitBreaker()
