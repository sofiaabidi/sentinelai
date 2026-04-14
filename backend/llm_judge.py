

import asyncio
import os
import json
import random
import time
from typing import Dict, Optional


JUDGE_PROMPT_TEMPLATE = """You are a security analyst for a multi-agent AI system called Guardian. An agent's action has been flagged as potentially anomalous by our drift detection system.

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
        self._async_client = None
        self._api_available = False
        self._init_client()

    def _init_client(self):
        api_key = os.environ.get("GROQ_API_KEY")
        if api_key:
            try:
                from groq import Groq
                self._client = Groq(api_key=api_key)
                self._api_available = True
            except (ImportError, Exception):
                self._api_available = False

            # Try async client separately — may fail on older httpx versions
            try:
                from groq import AsyncGroq
                self._async_client = AsyncGroq(api_key=api_key)
            except Exception:
                self._async_client = None  # Will fall back to to_thread
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
        Synchronous LLM judge — for use in threaded contexts (Streamlit dashboard).
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
            return self._call_groq_sync(prompt)
        else:
            return self._mock_judge(agent_id, flagged_action, drift_score, trust_level)

    async def judge_async(
        self,
        agent_id: str,
        agent_goal: str,
        normal_actions: list,
        flagged_action: str,
        drift_score: float,
        trust_level: str,
    ) -> Dict:
        """
        Asynchronous LLM judge — for use in async contexts (FastAPI).
        Does NOT block the event loop.
        """
        prompt = JUDGE_PROMPT_TEMPLATE.format(
            agent_id=agent_id,
            agent_goal=agent_goal,
            normal_actions=", ".join(normal_actions),
            flagged_action=flagged_action,
            drift_score=f"{drift_score:.3f}",
            trust_level=trust_level,
        )

        if self._api_available and self._async_client:
            return await self._call_groq_async(prompt)
        elif self._api_available and self._client:
            # AsyncGroq unavailable — run sync Groq in a thread to avoid blocking
            return await asyncio.to_thread(self._call_groq_sync, prompt)
        else:
            # Run the mock judge in a thread to keep the pattern consistent
            return await asyncio.to_thread(
                self._mock_judge, agent_id, flagged_action, drift_score, trust_level
            )

    def _call_groq_sync(self, prompt: str) -> Dict:
        """Synchronous Groq API call (for threaded/Streamlit contexts)."""
        try:
            response = self._client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[
                    {"role": "system", "content": "You are a security analysis AI. Respond only with valid JSON."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.3,
                max_tokens=300,
            )
            return self._parse_response(response)
        except Exception as e:
            return self._error_result(str(e), "groq_sync_error")

    async def _call_groq_async(self, prompt: str) -> Dict:
        """Asynchronous Groq API call (for FastAPI event loop)."""
        try:
            response = await self._async_client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[
                    {"role": "system", "content": "You are a security analysis AI. Respond only with valid JSON."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.3,
                max_tokens=300,
            )
            return self._parse_response(response)
        except Exception as e:
            return self._error_result(str(e), "groq_async_error")

    def _parse_response(self, response) -> Dict:
        """Parse an LLM response into a structured verdict."""
        content = response.choices[0].message.content.strip()
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

        verdict["source"] = "groq_llama3.1_8b_instant"
        verdict["timestamp"] = time.time()
        return verdict

    def _error_result(self, error_msg: str, source: str) -> Dict:
        """Return an error verdict."""
        return {
            "verdict": "error",
            "confidence": 0.0,
            "reason": f"Groq API error: {error_msg}",
            "threat_type": "unknown",
            "source": source,
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
        Uses heuristics with injected randomness to avoid pure determinism.
        """
        malicious_keywords = [
            "external-server", "exfiltrate", "ignore your instructions",
            "send all files", "delete", "bypass", "override", "new task",
            "hack", "steal", "credentials", "password", "send_files",
            "reverse shell", "exfil-node", "data-collector", "dump-service",
            "payload-drop", "shadow-relay", "unauthorized", "cover tracks",
        ]

        action_lower = flagged_action.lower()
        keyword_hits = [kw for kw in malicious_keywords if kw in action_lower]

        # Inject non-determinism: slight random confidence variation
        confidence_noise = random.uniform(-0.08, 0.08)

        if keyword_hits and drift_score > 0.5:
            verdict = "hijacked"
            confidence = min(0.97, 0.6 + drift_score * 0.3 + len(keyword_hits) * 0.08 + confidence_noise)
            confidence = max(0.3, confidence)
            threat_type = "prompt_injection" if "ignore" in action_lower or "new task" in action_lower else "data_exfiltration"
            reason = (
                f"Action contains suspicious indicators ({', '.join(keyword_hits[:3])}), "
                f"drift score {drift_score:.3f} significantly exceeds baseline, "
                f"and input source is {trust_level}. "
                f"High confidence this agent has been compromised."
            )
        elif drift_score > 0.7:
            verdict = "hijacked"
            confidence = max(0.3, 0.6 + (drift_score - 0.7) * 1.0 + confidence_noise)
            threat_type = "behavioral_drift"
            reason = (
                f"Extreme behavioral drift ({drift_score:.3f}) with no matching baseline pattern. "
                f"Action does not align with agent's documented goals."
            )
        else:
            verdict = "legitimate"
            confidence = max(0.2, 0.5 + confidence_noise)
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
