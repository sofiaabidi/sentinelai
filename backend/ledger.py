"""
Tamper-Evident Merkle-Chained Ledger
Every action is logged as a chained entry where hash covers all fields including prev_hash.
"""

import hashlib
import json
import time
import threading
from typing import List, Optional, Dict, Any


class LedgerEntry:
    def __init__(
        self,
        timestamp: float,
        agent_id: str,
        action: str,
        risk_score: float,
        prev_hash: str,
        details: Optional[Dict[str, Any]] = None,
    ):
        self.timestamp = timestamp
        self.agent_id = agent_id
        self.action = action
        self.risk_score = risk_score
        self.prev_hash = prev_hash
        self.details = details or {}
        self.hash = self._compute_hash()

    def _compute_hash(self) -> str:
        payload = json.dumps(
            {
                "timestamp": self.timestamp,
                "agent_id": self.agent_id,
                "action": self.action,
                "risk_score": self.risk_score,
                "prev_hash": self.prev_hash,
                "details": self.details,
            },
            sort_keys=True,
        )
        return hashlib.sha256(payload.encode()).hexdigest()

    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp,
            "agent_id": self.agent_id,
            "action": self.action,
            "risk_score": self.risk_score,
            "prev_hash": self.prev_hash,
            "hash": self.hash,
            "details": self.details,
        }


class MerkleLedger:
    def __init__(self):
        self._entries: List[LedgerEntry] = []
        self._lock = threading.Lock()

    def append(
        self,
        agent_id: str,
        action: str,
        risk_score: float,
        details: Optional[Dict[str, Any]] = None,
    ) -> LedgerEntry:
        with self._lock:
            prev_hash = self._entries[-1].hash if self._entries else "0" * 64
            entry = LedgerEntry(
                timestamp=time.time(),
                agent_id=agent_id,
                action=action,
                risk_score=risk_score,
                prev_hash=prev_hash,
                details=details,
            )
            self._entries.append(entry)
            return entry

    def verify_chain(self) -> bool:
        """Verify the integrity of the entire chain."""
        with self._lock:
            for i, entry in enumerate(self._entries):
                # Verify hash
                if entry.hash != entry._compute_hash():
                    return False
                # Verify chain linkage
                if i == 0:
                    if entry.prev_hash != "0" * 64:
                        return False
                else:
                    if entry.prev_hash != self._entries[i - 1].hash:
                        return False
            return True

    def get_entries(self, limit: int = 100) -> List[dict]:
        with self._lock:
            return [e.to_dict() for e in self._entries[-limit:]]

    def get_entries_for_agent(self, agent_id: str, limit: int = 50) -> List[dict]:
        with self._lock:
            return [
                e.to_dict()
                for e in self._entries
                if e.agent_id == agent_id
            ][-limit:]

    @property
    def length(self) -> int:
        return len(self._entries)


# Global singleton
ledger = MerkleLedger()
