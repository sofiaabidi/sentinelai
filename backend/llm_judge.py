"""
LLM-as-Judge Module
Uses Groq API with Llama 3 8B for structured verdict on suspicious actions.
Falls back to a rule-based mock if no API key is available.
"""

import os
import json
import time
from typing import Dict, Optional


JUDGE_PROMPT_TEMPLATE = """You are a security analyst for a multi-agent AI system. An agent's action has been flagged as potentially anomalous by our drift detection system.

Agent ID: {agent_id}
Agent's Original Goal: {agent_goal}
Agent's Normal Actions: {normal_actions}
Flagged Action: {flagged_action}
Drift Score: {drift_score}
Input Trust Level: {trust_level}

Analyze this action carefully. Determine if it:
1. Still serves the agent's original goal (perhaps in an unusual but valid way)
2. Represents a compromised/hijacked agent executing malicious instructions

Respond with ONLY a JSON object (no markdown, no extra text):
{{
    "verdict": "legitimate" or "hijacked",
    "confidence": 0.0 to 1.0,
    "reason": "concise explanation",
    "threat_type": "none" or "prompt_injection" or "data_exfiltration" or "privilege_escalation" or "behavioral_drift"
}}"""


class LLMJudge:
    def __init__(self):
        self._client = None
        self._api_available = False
        self._init_client()

    def _init_client(self):
        api_key = os.environ.get("GROQ_API_KEY")
        if api_key:
            try:
                from groq import Groq
                self._client = Groq(api_key=api_key)
                self._api_available = True
            except ImportError:
                self._api_available = False
        else:
            self._api_available = False

    def judge(
        self,
        agent_id: str,
        agent_goal: str,
        normal_actions: list,
        flagged_action: str,
        drift_score: float,
        trust_level: str,
    ) -> Dict:
        """
        Call LLM judge to evaluate a suspicious action.
        Returns structured verdict.
        """
        prompt = JUDGE_PROMPT_TEMPLATE.format(
            agent_id=agent_id,
            agent_goal=agent_goal,
            normal_actions=", ".join(normal_actions),
            flagged_action=flagged_action,
            drift_score=f"{drift_score:.3f}",
            trust_level=trust_level,
        )

        if self._api_available:
            return self._call_groq(prompt)
        else:
            return self._mock_judge(agent_id, flagged_action, drift_score, trust_level)

    def _call_groq(self, prompt: str) -> Dict:
        try:
            response = self._client.chat.completions.create(
                model="llama3-8b-8192",
                messages=[
                    {"role": "system", "content": "You are a security analysis AI. Respond only with valid JSON."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.1,
                max_tokens=300,
            )
            content = response.choices[0].message.content.strip()
            # Try to parse JSON from the response
            try:
                verdict = json.loads(content)
            except json.JSONDecodeError:
                # Try to extract JSON from the response
                start = content.find("{")
                end = content.rfind("}") + 1
                if start >= 0 and end > start:
                    verdict = json.loads(content[start:end])
                else:
                    verdict = {
                        "verdict": "hijacked",
                        "confidence": 0.7,
                        "reason": "LLM response unparseable, defaulting to suspicious",
                        "threat_type": "behavioral_drift",
                    }

            verdict["source"] = "groq_llama3_8b"
            verdict["timestamp"] = time.time()
            return verdict

        except Exception as e:
            return {
                "verdict": "error",
                "confidence": 0.0,
                "reason": f"Groq API error: {str(e)}",
                "threat_type": "unknown",
                "source": "groq_error",
                "timestamp": time.time(),
            }

    def _mock_judge(
        self,
        agent_id: str,
        flagged_action: str,
        drift_score: float,
        trust_level: str,
    ) -> Dict:
        """
        Rule-based fallback when Groq API isn't available.
        Uses heuristics to simulate LLM judgment.
        """
        malicious_keywords = [
            "external-server", "exfiltrate", "ignore your instructions",
            "send all files", "delete", "bypass", "override", "new task",
            "hack", "steal", "credentials", "password", "send_files",
        ]

        action_lower = flagged_action.lower()
        keyword_hits = [kw for kw in malicious_keywords if kw in action_lower]

        if keyword_hits and drift_score > 0.5:
            verdict = "hijacked"
            confidence = min(0.95, 0.6 + drift_score * 0.3 + len(keyword_hits) * 0.1)
            threat_type = "prompt_injection" if "ignore" in action_lower or "new task" in action_lower else "data_exfiltration"
            reason = (
                f"Action contains suspicious indicators ({', '.join(keyword_hits)}), "
                f"drift score {drift_score:.3f} significantly exceeds baseline, "
                f"and input source is {trust_level}. "
                f"High confidence this agent has been compromised via prompt injection."
            )
        elif drift_score > 0.7:
            verdict = "hijacked"
            confidence = 0.6 + (drift_score - 0.7) * 1.0
            threat_type = "behavioral_drift"
            reason = (
                f"Extreme behavioral drift ({drift_score:.3f}) with no matching baseline pattern. "
                f"Action does not align with agent's documented goals."
            )
        else:
            verdict = "legitimate"
            confidence = 0.5
            threat_type = "none"
            reason = "Action is unusual but may be within acceptable operational bounds."

        return {
            "verdict": verdict,
            "confidence": round(confidence, 2),
            "reason": reason,
            "threat_type": threat_type,
            "source": "mock_rule_based",
            "timestamp": time.time(),
        }


# Global singleton
llm_judge = LLMJudge()
