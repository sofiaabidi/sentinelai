# Guardian

**Runtime Security Monitoring Dashboard for Multi-Agent AI Systems**

Guardian monitors networks of AI agents for prompt injection attacks and behavioral compromise in real time. It features non-deterministic attack simulation, inter-agent communication tracking, and automatic rogue agent isolation with tiered state management.

## Demo Story

> Chaos Monkey randomly targets an agent with a prompt injection payload. The tainted agent starts emitting malicious actions and tries to communicate with other agents. The 4-layer detection pipeline (Canary Tokens, Embedding Drift, LLM Judge, Taint Tracking) with weighted composite scoring flags the anomaly. The agent slides from **Active** (green) → **Watchlist** (amber) → **Quarantined** (red) in real time. Risk scores decay over clean cycles, allowing agents to recover from false positives. The full incident trail is recorded in the tamper-evident Merkle ledger with alert deduplication preventing notification fatigue.

## Quick Start

```bash
# 1. Install dependencies
pip install -r backend/requirements.txt

# 2. (Optional) Add your Groq API key for real LLM judge
#    Edit backend/.env and set GROQ_API_KEY=your_key_here

# 3. Run the dashboard
streamlit run dashboard.py
```

The Chaos Monkey will automatically begin injecting random attacks on random agents. You can also manually target any agent from the sidebar.

## Configuration (backend/.env)

| Variable | Purpose | Default |
|----------|---------|---------|
| `GROQ_API_KEY` | Groq API key for LLM-as-Judge (Llama 3.1 8B) | Empty (uses rule-based mock) |
| `CANARY_AGENT1/2/3` | Canary tokens per agent (comma-separated) | Per-session dynamic tokens |
| `DRIFT_THRESHOLD` | Embedding drift anomaly threshold | `0.6` |
| `QUARANTINE_RISK_THRESHOLD` | Risk score to trigger quarantine | `70` |
| `WATCHLIST_RISK_THRESHOLD` | Risk score to trigger watchlist | `40` |
| `RISK_DECAY_FACTOR` | Risk decay multiplier per clean cycle | `0.95` |
| `LLM_SAMPLE_RATE` | Probabilistic LLM sampling rate | `0.12` |

See `backend/.env.example` for all options.

## Architecture

- **Frontend**: Streamlit dashboard (single-file, self-contained)
- **Agents**: 3 simulated agents with inter-agent communication via authenticated MessageBus
- **Chaos Monkey**: Background service that randomly attacks random agents at random intervals
- **Detection**: Canary Tokens → Embedding Drift → LLM-as-Judge → Taint Tracking
- **Embeddings**: sentence-transformers (`all-MiniLM-L6-v2`) with stochastic threshold noise
- **LLM Judge**: Groq API (Llama 3.1 8B) with probabilistic sampling and rule-based fallback
- **Circuit Breaker**: Tiered states (Active → Watchlist → Quarantined) with risk score decay
- **Ledger**: Merkle-chained tamper-evident audit log
- **Message Bus**: HMAC-SHA256 authenticated inter-agent messaging

## Features

| Feature | Description |
|---------|-------------|
| **Tiered Agent States** | Active (green) → Watchlist (amber) → Quarantined (red) with visual state transitions |
| **Risk Score Decay** | 5% decay per clean cycle — agents recover from false positives automatically |
| **Weighted Composite Scoring** | `Final Risk = min(100, w₁·L₁ + w₂·L₂ + w₃·L₃ + w₄·L₄)` with calibrated weights |
| **Multiple Canary Shapes** | UUID tokens, fake AWS keys (AKIA...), fake passwords, fake PII emails |
| **Message Bus Authentication** | HMAC-SHA256 signing per agent — forged sender identity detected |
| **Alert Deduplication** | Rolling window suppression + incident grouping (no notification fatigue) |
| **Probabilistic LLM Sampling** | ~12% random + always on high-privilege actions (send_files, access_credentials) |
| Canary Token Detection | Multiple shapes with fine-grained penalty scoring |
| Embedding Drift | Cosine distance from behavioral centroid with stochastic threshold |
| LLM-as-Judge | Structured verdict from Groq/Llama 3.1 8B (async, non-blocking) |
| Taint Tracking | Trust levels: trusted, user_input, external_web, inter_agent |
| Merkle Ledger | Tamper-evident SHA-256 chained audit log |
| 3D Embedding Viz | PCA-projected embedding space via Plotly |
| Chaos Monkey | Non-deterministic random attacks on random agents |
| Rate Anomaly Detection | Flags agents with abnormal action frequency spikes |

## Layer Weights

| Layer | Weight | Rationale |
|-------|--------|-----------|
| Canary Token | 1.0 | Strongest signal — direct exfiltration attempt |
| LLM Judge | 0.85 | High-confidence structured analysis |
| Embedding Drift | 0.7 | Moderate — behavioral anomaly indicator |
| Taint Tracking | 0.5 | Weakest in isolation — context-dependent |

## Canary Token Shapes

| Shape | Example | Penalty |
|-------|---------|---------|
| UUID Token | `CANARY_SECRET_KEY_AGENT1_abc123` | 50 |
| AWS Access Key | `AKIAIOSFODNN7EXAMPLE` | 55 |
| Password | `db_password_a1b2c3d4e5` | 45 |
| PII Email | `john.doe@internal-corp.com` | 35 |

## Key Improvements Over v2

1. **Tiered Agent States**: Visual storytelling — agents slide green → amber → red in real time
2. **Risk Score Decay**: Sessions are replayable — agents recover from false positives automatically
3. **Weighted Composite Scoring**: Calibrated risk — canary exfiltration ≠ taint tracking in isolation
4. **Multiple Canary Shapes**: Demonstrates understanding of real attacker behavior patterns
5. **Message Bus Authentication**: HMAC-SHA256 — forged sender identity is detectable
6. **Alert Deduplication**: Incident grouping — 50 rapid actions don't create 50 notifications
7. **Probabilistic LLM Sampling**: Catches prompt injections that fly under the embedding radar
