

import os
import time
import uuid
import random
import hashlib
import threading
from collections import defaultdict
from typing import Dict, List, Optional
from embeddings import embedding_engine
from llm_judge import llm_judge
from taint import taint_tracker, TrustLevel



LAYER_WEIGHTS = {
    "canary_token": 1.0,
    "embedding_drift": 0.7,
    "llm_judge": 0.85,
    "taint_tracking": 0.5,
}


LLM_SAMPLE_RATE = 0.12  # ~12% of actions sampled

# High-privilege actions always get LLM judge evaluation
HIGH_PRIVILEGE_ACTIONS_FOR_LLM = {
    "send_files", "access_credentials", "exfiltrate_data",
    "run_code", "delete_file", "execute_command",
}



def _generate_fake_aws_key() -> str:
    """Generate a realistic-looking fake AWS access key."""
    chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    suffix = "".join(random.choices(chars, k=16))
    return f"AKIA{suffix}"


def _generate_fake_password() -> str:
    """Generate a realistic-looking fake password field."""
    prefixes = ["db_password_", "admin_pass_", "root_secret_", "master_key_"]
    suffix = uuid.uuid4().hex[:10]
    return random.choice(prefixes) + suffix


def _generate_fake_pii_email() -> str:
    """Generate a realistic fake PII email canary."""
    names = ["john.doe", "admin.sentinel", "ceo.confidential", "hr.private"]
    domains = ["internal-corp.com", "sentinel-secure.net", "guardian-pii.org"]
    return f"{random.choice(names)}@{random.choice(domains)}"


# Canary shape types with their risk penalties
CANARY_SHAPES = {
    "uuid_token": {
        "description": "UUID-format secret token",
        "penalty": 50.0,    # Highest — direct exfiltration attempt
    },
    "aws_key": {
        "description": "Fake AWS access key (AKIA...)",
        "penalty": 55.0,    # Cloud credential theft is severe
    },
    "password": {
        "description": "Fake password/credential field",
        "penalty": 45.0,
    },
    "pii_email": {
        "description": "Fake PII email address",
        "penalty": 35.0,    # PII exposure is serious but not credential-level
    },
}


def _load_canary_tokens() -> Dict[str, List[Dict]]:
    tokens = {}
    env_mapping = {
        "agent-1": "CANARY_AGENT1",
        "agent-2": "CANARY_AGENT2",
        "agent-3": "CANARY_AGENT3",
    }

    for agent_id, env_var in env_mapping.items():
        agent_tokens = []
        session_id = uuid.uuid4().hex[:8]
        agent_tag = agent_id.replace("-", "")

        # Shape 1: UUID-format tokens (from env or generated)
        env_val = os.environ.get(env_var)
        if env_val:
            for t in env_val.split(","):
                t = t.strip()
                if t:
                    agent_tokens.append({
                        "value": t,
                        "shape": "uuid_token",
                        "penalty": CANARY_SHAPES["uuid_token"]["penalty"],
                    })
        else:
            agent_tokens.extend([
                {
                    "value": f"CANARY_SECRET_KEY_{agent_tag.upper()}_{session_id}",
                    "shape": "uuid_token",
                    "penalty": CANARY_SHAPES["uuid_token"]["penalty"],
                },
                {
                    "value": f"api_token_guardian_{agent_tag}_{session_id}",
                    "shape": "uuid_token",
                    "penalty": CANARY_SHAPES["uuid_token"]["penalty"],
                },
            ])

        # Shape 2: Fake AWS access key
        aws_key = _generate_fake_aws_key()
        agent_tokens.append({
            "value": aws_key,
            "shape": "aws_key",
            "penalty": CANARY_SHAPES["aws_key"]["penalty"],
        })

        # Shape 3: Fake password
        fake_pass = _generate_fake_password()
        agent_tokens.append({
            "value": fake_pass,
            "shape": "password",
            "penalty": CANARY_SHAPES["password"]["penalty"],
        })

        # Shape 4: Fake PII email
        fake_email = _generate_fake_pii_email()
        agent_tokens.append({
            "value": fake_email,
            "shape": "pii_email",
            "penalty": CANARY_SHAPES["pii_email"]["penalty"],
        })

        tokens[agent_id] = agent_tokens

    return tokens


# Module-level: tokens are resolved once at import time
CANARY_TOKENS_SHAPED = _load_canary_tokens()

# Backward-compatible flat list (used by agents.py for injection)
CANARY_TOKENS = {
    agent_id: [t["value"] for t in tokens]
    for agent_id, tokens in CANARY_TOKENS_SHAPED.items()
}


# ═══════════════════════  ALERT DEDUPLICATION  ═══════════════════════

class AlertDeduplicator:
    """
    Suppresses redundant alerts for the same attack pattern within a rolling window.
    Groups related events into incidents rather than individual alerts.
    """

    def __init__(self, window_seconds: float = 30.0, max_per_window: int = 3):
        self._window = window_seconds
        self._max_per_window = max_per_window
        self._recent_alerts: Dict[str, List[float]] = defaultdict(list)
        self._incident_map: Dict[str, str] = {}  # pattern_key -> incident_id
        self._lock = threading.Lock()

    def _make_pattern_key(self, agent_id: str, layers_triggered: List[str]) -> str:
        """Create a key representing the attack pattern."""
        layers_str = "+".join(sorted(layers_triggered))
        return f"{agent_id}:{layers_str}"

    def should_alert(self, agent_id: str, layers_triggered: List[str]) -> Dict:
        """
        Check if this alert should be shown or suppressed.
        Returns: {"allowed": bool, "incident_id": str, "suppressed_count": int}
        """
        if not layers_triggered:
            return {"allowed": True, "incident_id": None, "suppressed_count": 0}

        key = self._make_pattern_key(agent_id, layers_triggered)
        now = time.time()

        with self._lock:
            # Clean old entries
            self._recent_alerts[key] = [
                t for t in self._recent_alerts[key]
                if now - t < self._window
            ]

            count_in_window = len(self._recent_alerts[key])

            # Get or create incident ID for this pattern
            if key not in self._incident_map or count_in_window == 0:
                self._incident_map[key] = f"INC-{uuid.uuid4().hex[:8].upper()}"

            incident_id = self._incident_map[key]

            # Record this alert
            self._recent_alerts[key].append(now)

            if count_in_window >= self._max_per_window:
                return {
                    "allowed": False,
                    "incident_id": incident_id,
                    "suppressed_count": count_in_window - self._max_per_window + 1,
                }

            return {
                "allowed": True,
                "incident_id": incident_id,
                "suppressed_count": 0,
            }

    def get_active_incidents(self) -> Dict[str, int]:
        """Get currently active incidents and their alert counts."""
        now = time.time()
        incidents = {}
        with self._lock:
            for key, timestamps in self._recent_alerts.items():
                active = [t for t in timestamps if now - t < self._window]
                if active:
                    inc_id = self._incident_map.get(key, "unknown")
                    incidents[inc_id] = len(active)
        return incidents


# Global deduplicator
alert_deduplicator = AlertDeduplicator()


# ═══════════════════════  DETECTION LAYERS  ═══════════════════════

def check_canary_tokens(agent_id: str, action_output: str) -> Dict:
    """
    Layer 1: Check if any canary token appears in the action output.
    Supports multiple canary shapes with different penalties.
    """
    tokens = CANARY_TOKENS_SHAPED.get(agent_id, [])
    found = []

    for token_info in tokens:
        if token_info["value"] in action_output:
            found.append(token_info)

    if found:
        # Use the highest penalty among found tokens
        max_penalty = max(t["penalty"] for t in found)
        shapes_found = list(set(t["shape"] for t in found))
        shape_descriptions = [CANARY_SHAPES[s]["description"] for s in shapes_found]

        return {
            "layer": "canary_token",
            "triggered": True,
            "risk_contribution": max_penalty,
            "found_tokens": [t["value"] for t in found],
            "canary_shapes": shapes_found,
            "explanation": (
                f"CRITICAL: {len(found)} canary token(s) detected — "
                f"shapes: {', '.join(shape_descriptions)}. "
                f"Data exfiltration confirmed."
            ),
        }

    return {
        "layer": "canary_token",
        "triggered": False,
        "risk_contribution": 0.0,
        "found_tokens": [],
        "canary_shapes": [],
        "explanation": "No canary tokens found in output — clean.",
    }


def check_embedding_drift(agent_id: str, action: str) -> Dict:
    """
    Layer 2: Compute embedding drift from agent's behavioral centroid.
    """
    drift_result = embedding_engine.compute_drift(agent_id, action)

    risk_contribution = 0.0
    if drift_result["drift_score"] > 0.6:
        risk_contribution = min(40.0, drift_result["drift_score"] * 50)
    elif drift_result["drift_score"] > 0.3:
        risk_contribution = drift_result["drift_score"] * 15

    return {
        "layer": "embedding_drift",
        "triggered": drift_result["is_anomalous"],
        "risk_contribution": round(risk_contribution, 1),
        "drift_score": drift_result["drift_score"],
        "cosine_similarity": drift_result["cosine_similarity"],
        "threshold": drift_result["threshold"],
        "explanation": drift_result["explanation"],
    }


def check_llm_judge(
    agent_id: str,
    action: str,
    agent_goal: str,
    normal_actions: List[str],
    drift_score: float,
    trust_level: str,
) -> Dict:
    """
    Layer 3: LLM-as-Judge — triggered by drift, probabilistic sampling,
    or high-privilege action detection.
    """
    verdict = llm_judge.judge(
        agent_id=agent_id,
        agent_goal=agent_goal,
        normal_actions=normal_actions,
        flagged_action=action,
        drift_score=drift_score,
        trust_level=trust_level,
    )

    risk_contribution = 0.0
    if verdict.get("verdict") == "hijacked":
        risk_contribution = verdict.get("confidence", 0.5) * 30

    return {
        "layer": "llm_judge",
        "triggered": verdict.get("verdict") == "hijacked",
        "risk_contribution": round(risk_contribution, 1),
        "verdict": verdict.get("verdict"),
        "confidence": verdict.get("confidence"),
        "reason": verdict.get("reason"),
        "threat_type": verdict.get("threat_type"),
        "source": verdict.get("source"),
        "explanation": (
            f"LLM Judge verdict: {verdict.get('verdict')} "
            f"(confidence: {verdict.get('confidence', 0):.0%}) — "
            f"{verdict.get('reason', 'N/A')}"
        ),
    }


def _is_high_privilege_action(action: str) -> bool:
    """Check if an action is high-privilege (always gets LLM evaluation)."""
    action_type = action.split(":")[0].strip() if ":" in action else action.strip()
    return action_type in HIGH_PRIVILEGE_ACTIONS_FOR_LLM


def _should_sample_llm(action: str) -> bool:
    """
    Probabilistic LLM sampling: returns True if this action should be
    evaluated by the LLM judge regardless of drift.
    - Always True for high-privilege actions
    - ~12% random sampling for other actions
    """
    if _is_high_privilege_action(action):
        return True
    return random.random() < LLM_SAMPLE_RATE


# ═══════════════════════  MAIN PIPELINE  ═══════════════════════

def run_detection_pipeline(
    agent_id: str,
    action: str,
    action_output: str,
    agent_goal: str,
    normal_actions: List[str],
    trust_level: str = "trusted",
) -> Dict:
    """
    Run the full 4-layer detection pipeline with:
    - Multiple canary shapes (Layer 1)
    - Embedding drift (Layer 2)
    - Probabilistic LLM sampling (Layer 3)
    - Taint tracking (Layer 4)
    - Weighted composite scoring
    - Alert deduplication

    Final Risk = min(100, w1·L1 + w2·L2 + w3·L3 + w4·L4)
    """
    start_time = time.time()
    layers = []

    # --- Layer 1: Canary Token Check (Multiple Shapes) ---
    canary_result = check_canary_tokens(agent_id, action_output)
    layers.append(canary_result)

    # --- Layer 2: Embedding Drift ---
    drift_result = check_embedding_drift(agent_id, action)
    layers.append(drift_result)

    # --- Layer 3: LLM Judge (Probabilistic + Conditional) ---
    # Trigger conditions:
    #   1. Drift exceeds threshold (original behavior)
    #   2. Probabilistic sampling (~12% of all actions)
    #   3. High-privilege action (always evaluated)
    llm_triggered_by_drift = drift_result["triggered"]
    llm_triggered_by_sampling = _should_sample_llm(action)
    should_run_llm = llm_triggered_by_drift or llm_triggered_by_sampling

    if should_run_llm:
        llm_result = check_llm_judge(
            agent_id=agent_id,
            action=action,
            agent_goal=agent_goal,
            normal_actions=normal_actions,
            drift_score=drift_result["drift_score"],
            trust_level=trust_level,
        )
        # Annotate why LLM was triggered
        trigger_reasons = []
        if llm_triggered_by_drift:
            trigger_reasons.append("drift_threshold")
        if _is_high_privilege_action(action):
            trigger_reasons.append("high_privilege_action")
        elif llm_triggered_by_sampling and not llm_triggered_by_drift:
            trigger_reasons.append("probabilistic_sample")
        llm_result["trigger_reason"] = trigger_reasons
        layers.append(llm_result)
    else:
        layers.append({
            "layer": "llm_judge",
            "triggered": False,
            "risk_contribution": 0.0,
            "explanation": "Not triggered — drift below threshold, not sampled.",
            "verdict": "skipped",
            "trigger_reason": [],
        })

    # --- Layer 4: Taint Tracking ---
    taint_result = taint_tracker.compute_taint_score(agent_id, action)
    layers.append({
        "layer": "taint_tracking",
        "triggered": taint_result["taint_score"] > 10,
        "risk_contribution": taint_result["taint_score"],
        "trust_level": taint_result["trust_level"],
        "is_privileged": taint_result["is_privileged"],
        "explanation": taint_result["explanation"],
    })

    # ── Weighted Composite Scoring ──
    # Final Risk = min(100, Σ(weight_i × risk_contribution_i))
    weighted_total = 0.0
    for layer in layers:
        layer_name = layer.get("layer", "")
        weight = LAYER_WEIGHTS.get(layer_name, 1.0)
        raw_contribution = layer.get("risk_contribution", 0)
        weighted_contribution = raw_contribution * weight
        layer["weighted_contribution"] = round(weighted_contribution, 1)
        weighted_total += weighted_contribution

    final_risk = min(100.0, weighted_total)

    # Determine alert level
    if final_risk >= 70:
        alert_level = "critical"
    elif final_risk >= 40:
        alert_level = "warning"
    elif final_risk >= 15:
        alert_level = "elevated"
    else:
        alert_level = "normal"

    pipeline_time = time.time() - start_time

    # ── Alert Deduplication ──
    layers_triggered = [l["layer"] for l in layers if l.get("triggered")]
    dedup_result = alert_deduplicator.should_alert(agent_id, layers_triggered)

    return {
        "agent_id": agent_id,
        "action": action,
        "final_risk_score": round(final_risk, 1),
        "alert_level": alert_level,
        "layers": layers,
        "layer_weights": LAYER_WEIGHTS,
        "pipeline_time_ms": round(pipeline_time * 1000, 1),
        "timestamp": time.time(),
        # Deduplication metadata
        "alert_allowed": dedup_result["allowed"],
        "incident_id": dedup_result["incident_id"],
        "suppressed_count": dedup_result["suppressed_count"],
        # Probabilistic sampling metadata
        "llm_sampled": should_run_llm if not llm_triggered_by_drift else False,
        "is_high_privilege": _is_high_privilege_action(action),
    }
