# Guardian

**Runtime Security Monitoring Dashboard for Multi-Agent AI Systems**

Guardian monitors networks of AI agents for prompt injection attacks and behavioral compromise in real time. It features non-deterministic attack simulation, inter-agent communication tracking, and automatic rogue agent isolation.

## Demo Story

> Chaos Monkey randomly targets an agent with a prompt injection payload. The tainted agent starts emitting malicious actions and tries to communicate with other agents. The 3-layer detection pipeline (Canary Tokens, Embedding Drift, LLM Judge) flags the anomaly. The circuit breaker quarantines the rogue agent, blocking all its outbound messages. Other agents continue working normally. The full incident trail is recorded in the tamper-evident Merkle ledger.

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

See `backend/.env.example` for all options.

## Architecture

- **Frontend**: Streamlit dashboard (single-file, self-contained)
- **Agents**: 3 simulated agents with inter-agent communication via MessageBus
- **Chaos Monkey**: Background service that randomly attacks random agents at random intervals
- **Detection**: Canary Tokens -> Embedding Drift -> LLM-as-Judge -> Taint Tracking
- **Embeddings**: sentence-transformers (`all-MiniLM-L6-v2`) with stochastic threshold noise
- **LLM Judge**: Groq API (Llama 3.1 8B) with async support and rule-based fallback
- **Circuit Breaker**: Auto-quarantine at risk >= 70, with rate anomaly detection
- **Ledger**: Merkle-chained tamper-evident audit log

## Features

| Feature | Description |
|---------|-------------|
| Canary Token Detection | Dynamically loaded from .env or generated per-session |
| Embedding Drift | Cosine distance from behavioral centroid with stochastic threshold |
| LLM-as-Judge | Structured verdict from Groq/Llama 3.1 8B (async, non-blocking) |
| Taint Tracking | Trust levels: trusted, user_input, external_web, inter_agent |
| Circuit Breaker | Auto-quarantine at risk >= 70, with rate-limiting tripwire |
| Merkle Ledger | Tamper-evident SHA-256 chained audit log |
| 3D Embedding Viz | PCA-projected embedding space via Plotly |
| Inter-Agent Messaging | Real message bus between agents, dropped on quarantine |
| Chaos Monkey | Non-deterministic random attacks on random agents |
| Rate Anomaly Detection | Flags agents with abnormal action frequency spikes |

## Key Improvements Over v1

1. **Non-Deterministic**: Actions, attacks, timing, and drift thresholds all include randomness
2. **Inter-Agent Communication**: Real message passing between agents via MessageBus
3. **Chaos Monkey**: Automated random attacks instead of manual-only triggers
4. **Any Agent Can Go Rogue**: Not hardcoded to agent-1
5. **Async LLM Calls**: FastAPI event loop is never blocked
6. **Dynamic Canary Tokens**: Loaded from env or generated per-session
7. **Rate Limiting**: Abnormal action frequency triggers risk penalties
8. **Taint Tracking Fix**: Correctly identifies privileged action types
