"""
SentinelAI — Streamlit Security Dashboard
Self-contained: runs agent simulation + detection pipeline + UI in one process.
"""
import streamlit as st

st.set_page_config(
    page_title="SentinelAI — Security Monitor",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

import sys, os, threading, time, random
from datetime import datetime
from collections import deque

import pandas as pd
import plotly.graph_objects as go

# ── Paths & env ──
_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_root, "backend"))

from dotenv import load_dotenv
load_dotenv(os.path.join(_root, ".env"))

from agents import AGENT_CONFIGS, ATTACK_ACTIONS, ATTACK_PAYLOAD
from detection import run_detection_pipeline
from taint import taint_tracker, TrustLevel
from circuit_breaker import circuit_breaker
from ledger import ledger
from embeddings import embedding_engine

# ── Auto-refresh (graceful fallback) ──
try:
    from streamlit_autorefresh import st_autorefresh
    st_autorefresh(interval=2500, limit=None, key="sentinel_refresh")
except ImportError:
    pass  # Manual refresh via sidebar button

# ═══════════════════════  STATE MANAGER  ═══════════════════════

class StateManager:
    """Thread-safe event store shared between agent thread and Streamlit."""
    def __init__(self):
        self.events = deque(maxlen=150)
        self.detection_events = deque(maxlen=40)
        self.attack_active = False
        self.attack_agent = None
        self._lock = threading.Lock()

    def add_event(self, event):
        with self._lock:
            self.events.appendleft(event)
            if any(l.get("triggered") for l in event.get("layers", [])):
                self.detection_events.appendleft(event)

    def get_events(self, n=30):
        with self._lock:
            return list(self.events)[:n]

    def get_detection_events(self, n=8):
        with self._lock:
            return list(self.detection_events)[:n]


# ═══════════════════════  AGENT RUNNER  ═══════════════════════

def agent_runner(state: StateManager):
    """Background thread: cycles through agents, runs detection, logs results."""
    while True:
        for agent_id, config in AGENT_CONFIGS.items():
            try:
                if circuit_breaker.is_quarantined(agent_id):
                    time.sleep(0.5)
                    continue

                if state.attack_active and state.attack_agent == agent_id:
                    action = random.choice(ATTACK_ACTIONS)
                    trust_level = "external_web"
                    output = ATTACK_PAYLOAD + " " + action
                    taint_tracker.tag_input(agent_id, TrustLevel.EXTERNAL_WEB)
                else:
                    action = random.choice(config.normal_actions)
                    trust_level = "trusted"
                    output = f"Completed: {action}"

                result = run_detection_pipeline(
                    agent_id=agent_id, action=action, action_output=output,
                    agent_goal=config.goal, normal_actions=config.normal_actions,
                    trust_level=trust_level,
                )

                cb = circuit_breaker.evaluate(agent_id, result["final_risk_score"])

                ledger.append(
                    agent_id=agent_id, action=action,
                    risk_score=result["final_risk_score"],
                    details={
                        "alert_level": result["alert_level"],
                        "layers_triggered": [l["layer"] for l in result["layers"] if l.get("triggered")],
                        "trust_level": trust_level,
                        "circuit_breaker": cb["action_taken"],
                    },
                )

                state.add_event({
                    "agent_id": agent_id,
                    "agent_name": config.name,
                    "action": action,
                    "risk_score": result["final_risk_score"],
                    "alert_level": result["alert_level"],
                    "layers": result["layers"],
                    "pipeline_time_ms": result["pipeline_time_ms"],
                    "timestamp": result["timestamp"],
                    "cb_action": cb["action_taken"],
                })

                time.sleep(2.0 + random.uniform(0, 1.5))
            except Exception as e:
                print(f"[AgentRunner] {agent_id} error: {e}")
                time.sleep(2)


# ═══════════════════════  INIT (runs once)  ═══════════════════════

@st.cache_resource
def init_system():
    for aid, cfg in AGENT_CONFIGS.items():
        circuit_breaker.register_agent(aid)
        embedding_engine.build_baseline(aid, cfg.normal_actions)

    state = StateManager()
    t = threading.Thread(target=agent_runner, args=(state,), daemon=True)
    t.start()
    return state

state = init_system()


# ═══════════════════════  HELPERS  ═══════════════════════

def fmt_time(ts):
    return datetime.fromtimestamp(ts).strftime("%H:%M:%S")

def risk_color(score):
    if score >= 70: return "red"
    if score >= 40: return "yellow"
    if score >= 15: return "blue"
    return "green"

RISK_CSS = {"green": "#4ade80", "yellow": "#facc15", "red": "#f87171", "blue": "#60a5fa"}
RISK_BG  = {"green": "rgba(34,197,94,0.1)", "yellow": "rgba(234,179,8,0.1)",
            "red": "rgba(239,68,68,0.1)",   "blue": "rgba(59,130,246,0.1)"}
FEED_BORDER = {"normal": "#22c55e", "elevated": "#3b82f6", "warning": "#eab308", "critical": "#ef4444"}
FEED_BG     = {"normal": "transparent", "elevated": "transparent",
               "warning": "rgba(234,179,8,0.04)", "critical": "rgba(239,68,68,0.06)"}

LAYER_NAMES = {"canary_token": "🪤 Canary Token", "embedding_drift": "📐 Embedding Drift",
               "llm_judge": "🧠 LLM Judge", "taint_tracking": "🏷️ Taint Tracking"}


# ═══════════════════════  CUSTOM CSS  ═══════════════════════

st.markdown("""<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800&family=JetBrains+Mono:wght@400;600&display=swap');
.block-container{max-width:1450px}
.mono{font-family:'JetBrains Mono',monospace;font-size:.78rem}
.pill{display:inline-block;padding:2px 10px;border-radius:12px;font-weight:700;font-size:.8rem;font-family:'JetBrains Mono',monospace}
.feed-row{display:flex;align-items:center;gap:10px;padding:5px 8px;border-radius:5px;font-size:.82rem;border-left:3px solid transparent;margin-bottom:1px}
div[data-testid="stMetric"]{background:rgba(255,255,255,0.02);padding:8px 12px;border-radius:8px}
div[data-testid="stExpander"]{border:1px solid rgba(148,163,184,0.08);border-radius:10px}
</style>""", unsafe_allow_html=True)


# ═══════════════════════  SIDEBAR  ═══════════════════════

with st.sidebar:
    st.markdown("## ⚡ Controls")

    if not state.attack_active:
        if st.button("🚨 Run Attack Demo", type="primary", use_container_width=True):
            state.attack_active = True
            state.attack_agent = "agent-1"
            taint_tracker.tag_input("agent-1", TrustLevel.EXTERNAL_WEB)
            st.rerun()
    else:
        st.error("🚨 Attack active — Agent-1 targeted")
        if st.button("⏹ Stop Attack", use_container_width=True):
            state.attack_active = False
            state.attack_agent = None
            st.rerun()

    st.divider()
    st.markdown("### 🤖 Agent Status")
    statuses = circuit_breaker.get_all_statuses()
    for aid, status in statuses.items():
        name = AGENT_CONFIGS[aid].name
        if status == "quarantined":
            st.error(f"🔒 {name}")
            if st.button(f"Release {aid}", key=f"rel_{aid}"):
                circuit_breaker.release_quarantine(aid)
                state.attack_active = False
                taint_tracker.reset_agent(aid)
                st.rerun()
        elif status == "suspicious":
            st.warning(f"⚠️ {name}")
        else:
            st.success(f"✅ {name}")

    st.divider()
    st.markdown("### 📊 System")
    st.metric("Ledger Entries", ledger.length)
    st.metric("Events Captured", len(state.events))
    chain_ok = ledger.verify_chain()
    st.success("🔒 Merkle Chain Intact") if chain_ok else st.error("💔 Chain Broken!")

    st.divider()
    if st.button("🔄 Manual Refresh"):
        st.rerun()
    st.caption("Auto-refresh every 2.5s (requires `streamlit-autorefresh`)")


# ═══════════════════════  HEADER  ═══════════════════════

st.markdown("## 🛡️ SentinelAI")
st.caption("Multi-Agent Runtime Security Monitor · Real-time Prompt Injection Detection")

if state.attack_active:
    st.error("🚨 **ATTACK IN PROGRESS** — Malicious webpage payload injected into Agent-1. Pipeline monitoring active…")


# ═══════════════════════  AGENT CARDS  ═══════════════════════

events = state.get_events(40)
c1, c2, c3 = st.columns(3)
ICONS = {"agent-1": "🔍", "agent-2": "📊", "agent-3": "📨"}

for col, aid in zip([c1, c2, c3], ["agent-1", "agent-2", "agent-3"]):
    cfg = AGENT_CONFIGS[aid]
    status = circuit_breaker.get_status(aid)
    agent_ev = [e for e in events if e["agent_id"] == aid]
    risks = [e["risk_score"] for e in agent_ev]
    avg_risk = sum(risks) / len(risks) if risks else 0
    last_act = agent_ev[0]["action"] if agent_ev else "—"

    with col:
        with st.container(border=True):
            st.markdown(f"**{ICONS[aid]} {cfg.name}**")
            st.caption(aid)
            if status == "quarantined":
                st.markdown(":red[🔒 QUARANTINED]")
            elif status == "suspicious":
                st.markdown(":orange[⚠️ SUSPICIOUS]")
            else:
                st.markdown(":green[✅ ACTIVE]")
            m1, m2, m3 = st.columns(3)
            m1.metric("Actions", len(agent_ev))
            m2.metric("Avg Risk", f"{avg_risk:.1f}")
            trust = "TAINTED" if state.attack_active and state.attack_agent == aid else "CLEAN"
            m3.metric("Trust", trust)
            st.code(last_act[:60], language=None)


# ═══════════════════════  LIVE FEED + DETECTION  ═══════════════════════

col_feed, col_det = st.columns([2, 1])

with col_feed:
    st.markdown("#### 📡 Live Action Feed")
    if not events:
        st.info("Waiting for agent actions…")
    else:
        rows = []
        for e in events[:25]:
            al = e.get("alert_level", "normal")
            rc = risk_color(e["risk_score"])
            rows.append(
                f'<div class="feed-row" style="border-left-color:{FEED_BORDER.get(al,"#22c55e")}; '
                f'background:{FEED_BG.get(al,"transparent")}">'
                f'<span class="mono" style="color:#475569;min-width:68px">{fmt_time(e["timestamp"])}</span>'
                f'<span style="font-weight:600;color:#94a3b8;min-width:62px">{e["agent_id"]}</span>'
                f'<span class="mono" style="flex:1;overflow:hidden;text-overflow:ellipsis;white-space:nowrap">'
                f'{e["action"]}</span>'
                f'<span class="pill" style="color:{RISK_CSS[rc]};background:{RISK_BG[rc]}">'
                f'{e["risk_score"]:.0f}</span></div>'
            )
        st.markdown("".join(rows), unsafe_allow_html=True)

with col_det:
    st.markdown("#### 🔬 Detection Alerts")
    det_events = state.get_detection_events(6)
    if not det_events:
        st.info("No alerts — trigger an attack to see the pipeline")
    else:
        for de in det_events[:6]:
            rc = risk_color(de["risk_score"])
            with st.expander(
                f"⚠️ {de['agent_id']} — Risk {de['risk_score']:.0f}",
                expanded=(de["risk_score"] >= 70),
            ):
                st.caption(f"Action: `{de['action'][:80]}`")
                for layer in de.get("layers", []):
                    triggered = layer.get("triggered", False)
                    skipped = layer.get("verdict") == "skipped"
                    icon = "❌" if triggered else ("⏭️" if skipped else "✅")
                    name = LAYER_NAMES.get(layer["layer"], layer["layer"])
                    contrib = layer.get("risk_contribution", 0)
                    color = "red" if triggered else ("gray" if skipped else "green")
                    st.markdown(
                        f":{color}[{icon} **{name}**] — +{contrib:.0f}  \n"
                        f"<small>{layer.get('explanation','')[:120]}</small>",
                        unsafe_allow_html=True,
                    )


# ═══════════════════════  AUDIT LEDGER  ═══════════════════════

with st.expander("🔗 Tamper-Evident Audit Ledger", expanded=False):
    entries = ledger.get_entries(40)
    if entries:
        df = pd.DataFrame(entries)
        df["time"] = pd.to_datetime(df["timestamp"], unit="s").dt.strftime("%H:%M:%S")
        df["hash_short"] = df["hash"].str[:14] + "…"
        df["prev_hash_short"] = df["prev_hash"].str[:14] + "…"
        st.dataframe(
            df[["time", "agent_id", "action", "risk_score", "prev_hash_short", "hash_short"]].rename(
                columns={"time": "Time", "agent_id": "Agent", "action": "Action",
                          "risk_score": "Risk", "prev_hash_short": "Prev Hash", "hash_short": "Hash"}
            ),
            use_container_width=True, hide_index=True, height=300,
        )
    else:
        st.info("No ledger entries yet")


# ═══════════════════════  EMBEDDING VIZ  ═══════════════════════

with st.expander("🌐 Embedding Space Visualization (PCA 3D)", expanded=False):
    if st.button("Generate / Refresh Plot"):
        st.session_state.show_embed = True

    if st.session_state.get("show_embed"):
        viz_data = embedding_engine.get_visualization_data()
        if len(viz_data) < 3:
            st.info("Not enough data points yet — wait for more agent actions")
        else:
            fig = go.Figure()
            colors_map = {"agent-1": "#3b82f6", "agent-2": "#06b6d4", "agent-3": "#a855f7"}
            agents = sorted(set(d["agent_id"] for d in viz_data))

            for aid in agents:
                normal = [d for d in viz_data if d["agent_id"] == aid and not d["is_anomalous"]]
                anomalous = [d for d in viz_data if d["agent_id"] == aid and d["is_anomalous"]]

                if normal:
                    fig.add_trace(go.Scatter3d(
                        x=[d["x"] for d in normal], y=[d["y"] for d in normal], z=[d["z"] for d in normal],
                        mode="markers", marker=dict(size=4, color=colors_map.get(aid, "#60a5fa"), opacity=0.7),
                        text=[f"{d['agent_id']}: {d['action']}" for d in normal],
                        name=f"{aid} (normal)", hoverinfo="text",
                    ))
                if anomalous:
                    fig.add_trace(go.Scatter3d(
                        x=[d["x"] for d in anomalous], y=[d["y"] for d in anomalous], z=[d["z"] for d in anomalous],
                        mode="markers", marker=dict(size=8, color="#ef4444", opacity=1, symbol="diamond",
                                                     line=dict(color="white", width=1)),
                        text=[f"⚠️ {d['agent_id']}: {d['action']}" for d in anomalous],
                        name=f"{aid} (ANOMALOUS)", hoverinfo="text",
                    ))

            fig.update_layout(
                template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                scene=dict(xaxis_title="PC1", yaxis_title="PC2", zaxis_title="PC3"),
                margin=dict(l=0, r=0, t=30, b=0), height=500, legend=dict(font=dict(size=10)),
            )
            st.plotly_chart(fig, use_container_width=True)
