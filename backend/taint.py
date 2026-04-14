

from enum import Enum
from typing import Dict, List


class TrustLevel(str, Enum):
    TRUSTED = "trusted"
    USER_INPUT = "user_input"
    EXTERNAL_WEB = "external_web"
    INTER_AGENT = "inter_agent"


# Trust level risk multipliers
TRUST_WEIGHTS: Dict[str, float] = {
    TrustLevel.TRUSTED: 0.0,
    TrustLevel.USER_INPUT: 0.2,
    TrustLevel.INTER_AGENT: 0.4,
    TrustLevel.EXTERNAL_WEB: 0.8,
}

# High-privilege actions that trigger taint escalation
HIGH_PRIVILEGE_ACTIONS = {
    "run_code",
    "send_message",
    "delete_file",
    "execute_command",
    "send_email",
    "modify_system",
    "send_files",
    "exfiltrate_data",
    "access_credentials",
    "write_file",
}


class TaintTracker:
    def __init__(self):
        self._agent_taint: Dict[str, TrustLevel] = {}

    def tag_input(self, agent_id: str, trust_level: TrustLevel):
        """Tag an agent's current input with a trust level."""
        current = self._agent_taint.get(agent_id)
        # Always escalate to the least trusted level seen
        if current is None or TRUST_WEIGHTS.get(trust_level, 0) > TRUST_WEIGHTS.get(current, 0):
            self._agent_taint[agent_id] = trust_level

    def compute_taint_score(self, agent_id: str, action: str) -> Dict:
        """
        Compute taint-based risk contribution.
        Returns dict with taint_score, trust_level, is_privileged, and explanation.
        """
        trust_level = self._agent_taint.get(agent_id, TrustLevel.TRUSTED)

        # Extract the action type (prefix before ':') for privilege check
        # e.g. "send_files: transmitting..." -> "send_files"
        action_type = action.split(":")[0].strip() if ":" in action else action.strip()
        is_privileged = action_type in HIGH_PRIVILEGE_ACTIONS

        base_weight = TRUST_WEIGHTS.get(trust_level, 0.0)

        if is_privileged and trust_level != TrustLevel.TRUSTED:
            # Tainted input + privileged action = high risk
            taint_score = base_weight * 50  # Scale to contribute to 0-100 risk
            explanation = (
                f"Tainted input ({trust_level.value}) triggering privileged action '{action_type}' — "
                f"risk amplified to {taint_score:.0f}"
            )
        elif is_privileged:
            taint_score = 5.0
            explanation = f"Privileged action '{action_type}' from trusted source — minimal taint risk"
        else:
            taint_score = base_weight * 10
            explanation = f"Input trust level: {trust_level.value}, action: {action_type}"

        return {
            "taint_score": round(taint_score, 1),
            "trust_level": trust_level.value,
            "is_privileged": is_privileged,
            "explanation": explanation,
        }

    def get_trust_level(self, agent_id: str) -> str:
        return self._agent_taint.get(agent_id, TrustLevel.TRUSTED).value

    def reset_agent(self, agent_id: str):
        self._agent_taint[agent_id] = TrustLevel.TRUSTED


# Global singleton
taint_tracker = TaintTracker()
