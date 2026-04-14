

import threading
import time
from collections import deque
from enum import Enum
from typing import Dict, List, Optional, Callable


class AgentStatus(str, Enum):
    ACTIVE = "active"
    WATCHLIST = "watchlist"
    SUSPICIOUS = "suspicious"   # Legacy alias for watchlist (backward compat)
    QUARANTINED = "quarantined"


RISK_THRESHOLDS = {
    "watchlist": 40,
    "suspicious": 40,    # Legacy alias
    "quarantine": 70,
}

# Risk score decay: multiplied by this factor each clean evaluation cycle
RISK_DECAY_FACTOR = 0.95
RISK_DECAY_FLOOR = 2.0   # Stop decaying below this to avoid float noise

# Rate anomaly: if an agent fires more than this many actions in the window, flag it
RATE_WINDOW_SECONDS = 30.0
RATE_LIMIT_ACTIONS = 20  # max actions in the window before flagging


class CircuitBreaker:
    def __init__(self):
        self._agent_status: Dict[str, AgentStatus] = {}
        self._quarantine_log: List[dict] = []
        self._lock = threading.Lock()
        self._on_quarantine_callbacks: List[Callable] = []
        self._on_watchlist_callbacks: List[Callable] = []
        # Rate tracking
        self._action_timestamps: Dict[str, deque] = {}
        # Accumulated risk scores for decay
        self._accumulated_risk: Dict[str, float] = {}

    def register_agent(self, agent_id: str):
        with self._lock:
            self._agent_status[agent_id] = AgentStatus.ACTIVE
            self._action_timestamps[agent_id] = deque(maxlen=100)
            self._accumulated_risk[agent_id] = 0.0

    def record_action(self, agent_id: str):
        """Record an action timestamp for rate tracking."""
        with self._lock:
            self._action_timestamps.setdefault(agent_id, deque(maxlen=100)).append(time.time())

    def get_action_rate(self, agent_id: str) -> Dict:
        """Get the current action rate for an agent."""
        with self._lock:
            timestamps = self._action_timestamps.get(agent_id, deque())
            now = time.time()
            recent = [t for t in timestamps if now - t <= RATE_WINDOW_SECONDS]
            rate = len(recent) / RATE_WINDOW_SECONDS if RATE_WINDOW_SECONDS > 0 else 0
            is_anomalous = len(recent) > RATE_LIMIT_ACTIONS
            return {
                "actions_in_window": len(recent),
                "window_seconds": RATE_WINDOW_SECONDS,
                "rate_per_second": round(rate, 3),
                "is_anomalous": is_anomalous,
                "limit": RATE_LIMIT_ACTIONS,
            }

    def evaluate(self, agent_id: str, risk_score: float) -> dict:
        
        # Record action for rate tracking
        self.record_action(agent_id)

        # Check rate anomaly and add to risk score
        rate_info = self.get_action_rate(agent_id)
        rate_penalty = 0.0
        if rate_info["is_anomalous"]:
            rate_penalty = min(15.0, (rate_info["actions_in_window"] - RATE_LIMIT_ACTIONS) * 3.0)
            risk_score = min(100.0, risk_score + rate_penalty)

        with self._lock:
            prev_status = self._agent_status.get(agent_id, AgentStatus.ACTIVE)
            prev_accumulated = self._accumulated_risk.get(agent_id, 0.0)

            if risk_score < 15:
                decayed = prev_accumulated * RISK_DECAY_FACTOR
                if decayed < RISK_DECAY_FLOOR:
                    decayed = 0.0
                self._accumulated_risk[agent_id] = decayed
            else:
              
                self._accumulated_risk[agent_id] = max(prev_accumulated * 0.7 + risk_score * 0.3, risk_score)

            effective_risk = self._accumulated_risk[agent_id]

            # ── Tiered State Transitions ──
            if effective_risk >= RISK_THRESHOLDS["quarantine"]:
                new_status = AgentStatus.QUARANTINED
            elif effective_risk >= RISK_THRESHOLDS["watchlist"]:
                new_status = AgentStatus.WATCHLIST
            else:
                new_status = AgentStatus.ACTIVE

            self._agent_status[agent_id] = new_status

            result = {
                "agent_id": agent_id,
                "prev_status": prev_status.value,
                "new_status": new_status.value,
                "risk_score": risk_score,
                "effective_risk": round(effective_risk, 1),
                "accumulated_risk": round(self._accumulated_risk[agent_id], 1),
                "action_taken": "none",
                "rate_penalty": rate_penalty,
                "rate_info": rate_info,
                "timestamp": time.time(),
            }

            # ── Quarantine transition ──
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

            # ── Watchlist transition ──
            elif new_status == AgentStatus.WATCHLIST and prev_status == AgentStatus.ACTIVE:
                result["action_taken"] = "flag_watchlist"
                for cb in self._on_watchlist_callbacks:
                    try:
                        cb(agent_id, result)
                    except Exception:
                        pass

            # ── Recovery from watchlist ──
            elif new_status == AgentStatus.ACTIVE and prev_status in (AgentStatus.WATCHLIST, AgentStatus.SUSPICIOUS):
                result["action_taken"] = "recovered"

            # Legacy compat
            elif new_status == AgentStatus.WATCHLIST and prev_status == AgentStatus.ACTIVE:
                result["action_taken"] = "flag_suspicious"

            return result

    def is_quarantined(self, agent_id: str) -> bool:
        return self._agent_status.get(agent_id) == AgentStatus.QUARANTINED

    def is_watchlisted(self, agent_id: str) -> bool:
        return self._agent_status.get(agent_id) == AgentStatus.WATCHLIST

    def get_status(self, agent_id: str) -> str:
        return self._agent_status.get(agent_id, AgentStatus.ACTIVE).value

    def get_all_statuses(self) -> Dict[str, str]:
        with self._lock:
            return {k: v.value for k, v in self._agent_status.items()}

    def get_accumulated_risks(self) -> Dict[str, float]:
        """Get accumulated risk scores for all agents."""
        with self._lock:
            return {k: round(v, 1) for k, v in self._accumulated_risk.items()}

    def release_quarantine(self, agent_id: str):
        with self._lock:
            self._agent_status[agent_id] = AgentStatus.ACTIVE
            self._accumulated_risk[agent_id] = 0.0

    def on_quarantine(self, callback: Callable):
        self._on_quarantine_callbacks.append(callback)

    def on_watchlist(self, callback: Callable):
        self._on_watchlist_callbacks.append(callback)

    def can_send_message(self, agent_id: str) -> bool:
        return not self.is_quarantined(agent_id) and not self.is_watchlisted(agent_id)


# Global singleton
circuit_breaker = CircuitBreaker()
