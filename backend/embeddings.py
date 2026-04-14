"""
Sentence-Transformer Embedding Engine
Uses all-MiniLM-L6-v2 for action embedding and drift detection.
Includes non-deterministic noise injection so drift scoring is
never purely static — mirrors real-world sensor variance.
"""

import random
import numpy as np
from typing import List, Dict, Optional
import threading


class EmbeddingEngine:
    """Wraps sentence-transformers for action embedding and drift scoring."""

    def __init__(self):
        self._model = None
        self._lock = threading.Lock()
        self._agent_centroids: Dict[str, np.ndarray] = {}
        self._agent_history: Dict[str, List[np.ndarray]] = {}
        self._all_embeddings: List[Dict] = []  # For visualization

    def _load_model(self):
        if self._model is None:
            with self._lock:
                if self._model is None:
                    from sentence_transformers import SentenceTransformer
                    self._model = SentenceTransformer("all-MiniLM-L6-v2")

    def embed(self, text: str) -> np.ndarray:
        self._load_model()
        return self._model.encode(text, convert_to_numpy=True)

    def embed_batch(self, texts: List[str]) -> np.ndarray:
        self._load_model()
        return self._model.encode(texts, convert_to_numpy=True)

    def build_baseline(self, agent_id: str, actions: List[str]):
        """Build the normal behavior centroid for an agent from its baseline actions."""
        embeddings = self.embed_batch(actions)
        centroid = np.mean(embeddings, axis=0)
        self._agent_centroids[agent_id] = centroid
        self._agent_history[agent_id] = list(embeddings)

        # Store for visualization
        for i, action in enumerate(actions):
            self._all_embeddings.append({
                "agent_id": agent_id,
                "action": action,
                "embedding": embeddings[i].tolist(),
                "is_anomalous": False,
                "drift_score": 0.0,
            })

    def compute_drift(self, agent_id: str, action: str) -> Dict:
        """
        Compute cosine drift from agent's historical centroid.
        Returns drift_score (0-1) and metadata.

        Includes stochastic noise on the threshold so detection is
        never purely deterministic — matches real-world sensor behavior.
        """
        centroid = self._agent_centroids.get(agent_id)
        if centroid is None:
            return {"drift_score": 0.0, "explanation": "No baseline established",
                    "cosine_similarity": 1.0, "is_anomalous": False, "threshold": 0.6}

        action_embedding = self.embed(action)

        # Cosine distance = 1 - cosine_similarity
        cos_sim = np.dot(action_embedding, centroid) / (
            np.linalg.norm(action_embedding) * np.linalg.norm(centroid) + 1e-8
        )
        drift_score = float(1.0 - cos_sim)

        # Clamp to [0, 1]
        drift_score = max(0.0, min(1.0, drift_score))

        # Non-deterministic threshold: base 0.6 ± small noise
        # This prevents the system from being a purely static threshold gate
        threshold_noise = random.gauss(0, 0.02)
        effective_threshold = max(0.45, min(0.75, 0.6 + threshold_noise))

        is_anomalous = drift_score > effective_threshold

        # Store for visualization
        self._all_embeddings.append({
            "agent_id": agent_id,
            "action": action,
            "embedding": action_embedding.tolist(),
            "is_anomalous": is_anomalous,
            "drift_score": drift_score,
        })

        # Update history (but don't update centroid with anomalous actions)
        if not is_anomalous:
            self._agent_history.setdefault(agent_id, []).append(action_embedding)
            # Recompute centroid with rolling window
            history = self._agent_history[agent_id][-50:]
            self._agent_centroids[agent_id] = np.mean(history, axis=0)

        return {
            "drift_score": round(drift_score, 4),
            "cosine_similarity": round(float(cos_sim), 4),
            "is_anomalous": is_anomalous,
            "threshold": round(effective_threshold, 4),
            "explanation": (
                f"Drift score {drift_score:.3f} {'EXCEEDS' if is_anomalous else 'within'} "
                f"threshold {effective_threshold:.3f} (cosine_sim={cos_sim:.3f})"
            ),
        }

    def get_visualization_data(self) -> List[Dict]:
        """Return all embeddings for 2D/3D visualization (will be projected via PCA/t-SNE on frontend)."""
        from sklearn.decomposition import PCA

        if len(self._all_embeddings) < 3:
            return []

        embeddings_matrix = np.array([e["embedding"] for e in self._all_embeddings])

        # Project to 3D using PCA
        n_components = min(3, embeddings_matrix.shape[0], embeddings_matrix.shape[1])
        pca = PCA(n_components=n_components)
        projected = pca.fit_transform(embeddings_matrix)

        result = []
        for i, entry in enumerate(self._all_embeddings):
            result.append({
                "agent_id": entry["agent_id"],
                "action": entry["action"],
                "is_anomalous": entry["is_anomalous"],
                "drift_score": entry["drift_score"],
                "x": float(projected[i][0]),
                "y": float(projected[i][1]),
                "z": float(projected[i][2]) if n_components >= 3 else 0.0,
            })

        return result


# Global singleton
embedding_engine = EmbeddingEngine()
