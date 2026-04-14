# Gaurdian 🛡️

**Runtime Security Monitoring Dashboard for Multi-Agent AI Systems**

SentinelAI monitors networks of AI agents for prompt injection attacks and behavioral compromise in real time.

## Demo Story

> Malicious webpage → Agent-1 tainted → drift detected → LLM judge confirms hijack → quarantine fires → Agent-2 and Agent-3 never receive the compromised instruction → full audit trail in the ledger

## Quick Start

```bash
# 1. Install dependencies
pip install -r backend/requirements.txt

# 2. (Optional) Add your Groq API key for real LLM judge
#    Edit .env and set GROQ_API_KEY=your_key_here

# 3. Run the dashboard
streamlit run dashboard.py
```

Then click **🚨 Run Attack Demo** in the sidebar.

## Configuration (.env)

| Variable | Purpose | Default |
|----------|---------|---------|
| `GROQ_API_KEY` | Groq API key for LLM-as-Judge (Llama 3 8B) | Empty (uses rule-based mock) |
| `CANARY_AGENT1/2/3` | Canary tokens per agent (comma-separated) | Built-in fake secrets |
| `DRIFT_THRESHOLD` | Embedding drift anomaly threshold | `0.6` |
| `QUARANTINE_RISK_THRESHOLD` | Risk score to trigger quarantine | `70` |

See `.env.example` for all options.

## Architecture

- **Frontend**: Streamlit dashboard (single-file, self-contained)
- **Agents**: 3 simulated agents running in a background thread
- **Detection**: Canary Tokens → Embedding Drift → LLM-as-Judge → Taint Tracking
- **Embeddings**: sentence-transformers (`all-MiniLM-L6-v2`)
- **LLM Judge**: Groq API (Llama 3 8B) with rule-based fallback
- **Ledger**: Merkle-chained tamper-evident audit log

## Features

| Feature | Description |
|---------|-------------|
| 🪤 Canary Token Detection | Fake sensitive strings planted in agent contexts |
| 📐 Embedding Drift | Cosine distance from behavioral centroid (threshold: 0.6) |
| 🧠 LLM-as-Judge | Structured verdict from Groq/Llama 3 8B |
| 🏷️ Taint Tracking | Trust levels: trusted, user_input, external_web, inter_agent |
| 🔒 Circuit Breaker | Auto-quarantine at risk ≥ 70 |
| 🔗 Merkle Ledger | Tamper-evident SHA-256 chained audit log |
| 🌐 3D Embedding Viz | PCA-projected embedding space via Plotly |
