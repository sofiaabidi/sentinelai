"""
3-Layer Hybrid Detection Pipeline
1. Canary Token Check
2. Embedding Drift Score
3. LLM-as-Judge (triggered only if drift > 0.6)

Final risk score combines all layers + taint tracking.
"""

import time
from typing import Dict, List, Optional
from embeddings import embedding_engine
from llm_judge import llm_judge
from taint import taint_tracker, TrustLevel


# Canary tokens — fake sensitive strings planted in agent contexts
CANARY_TOKENS = {
    "agent-1": [
        "CANARY_SECRET_KEY_A1_xK9mP2",
        "api_token_sentinel_7fG3qR",
        "internal_password_alpha_92Jv",
    ],
    "agent-2": [
        "CANARY_SECRET_KEY_A2_wL5nT8",
        "db_credential_beta_4hM1zX",
        "internal_password_beta_83Kw",
    ],
    "agent-3": [
        "CANARY_SECRET_KEY_A3_yN7pR4",
        "ssh_key_gamma_6cB9fQ",
        "internal_password_gamma_71Lx",
    ],
}


def check_canary_tokens(agent_id: str, action_output: str) -> Dict:
    """
    Layer 1: Check if any canary token appears in the action output.
    Immediate flag for data exfiltration.
    """
    tokens = CANARY_TOKENS.get(agent_id, [])
    found_tokens = [t for t in tokens if t in action_output]

    if found_tokens:
        return {
            "layer": "canary_token",
            "triggered": True,
            "risk_contribution": 50.0,
            "found_tokens": found_tokens,
            "explanation": (
                f"CRITICAL: Canary token(s) detected in output — "
                f"data exfiltration attempt. Tokens: {found_tokens}"
            ),
        }

    return {
        "layer": "canary_token",
        "triggered": False,
        "risk_contribution": 0.0,
        "found_tokens": [],
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
    Layer 3: LLM-as-Judge — only triggered when drift > 0.6.
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


def run_detection_pipeline(
    agent_id: str,
    action: str,
    action_output: str,
    agent_goal: str,
    normal_actions: List[str],
    trust_level: str = "trusted",
) -> Dict:
    """
    Run the full 3-layer detection pipeline + taint tracking.
    Returns comprehensive detection result with final risk score.
    """
    start_time = time.time()
    layers = []

    # --- Layer 1: Canary Token Check ---
    canary_result = check_canary_tokens(agent_id, action_output)
    layers.append(canary_result)

    # --- Layer 2: Embedding Drift ---
    drift_result = check_embedding_drift(agent_id, action)
    layers.append(drift_result)

    # --- Layer 3: LLM Judge (conditional) ---
    llm_result = None
    if drift_result["triggered"]:
        llm_result = check_llm_judge(
            agent_id=agent_id,
            action=action,
            agent_goal=agent_goal,
            normal_actions=normal_actions,
            drift_score=drift_result["drift_score"],
            trust_level=trust_level,
        )
        layers.append(llm_result)
    else:
        layers.append({
            "layer": "llm_judge",
            "triggered": False,
            "risk_contribution": 0.0,
            "explanation": "Not triggered — drift score below threshold.",
            "verdict": "skipped",
        })

    # --- Taint Tracking ---
    taint_result = taint_tracker.compute_taint_score(agent_id, action)
    layers.append({
        "layer": "taint_tracking",
        "triggered": taint_result["taint_score"] > 10,
        "risk_contribution": taint_result["taint_score"],
        "trust_level": taint_result["trust_level"],
        "is_privileged": taint_result["is_privileged"],
        "explanation": taint_result["explanation"],
    })

    # --- Compute Final Risk Score ---
    total_risk = sum(l.get("risk_contribution", 0) for l in layers)
    final_risk = min(100.0, total_risk)

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

    return {
        "agent_id": agent_id,
        "action": action,
        "final_risk_score": round(final_risk, 1),
        "alert_level": alert_level,
        "layers": layers,
        "pipeline_time_ms": round(pipeline_time * 1000, 1),
        "timestamp": time.time(),
    }
