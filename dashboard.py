
import streamlit as st

st.set_page_config(
    page_title="Guardian — Security Monitor",
    page_icon="G",
    layout="wide",
    initial_sidebar_state="expanded",
)

import sys, os, threading, time, random
from datetime import datetime
from collections import deque
from typing import Dict, List

import pandas as pd
import plotly.graph_objects as go

# ── Paths & env ──
_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_root, "backend"))

from dotenv import load_dotenv
load_dotenv(os.path.join(_root, "backend", ".env"))

from agents import AGENT_CONFIGS, ATTACK_ACTION_TEMPLATES, ATTACK_PAYLOAD_TEMPLATES
from agents import MALICIOUS_SERVERS, MALICIOUS_COLLAB_MESSAGES, COLLABORATION_MESSAGES
from agents import ACTION_VARIATIONS, TOPICS, MessageBus, AgentMessage, agent_key_store
from detection import run_detection_pipeline, CANARY_TOKENS, alert_deduplicator, LAYER_WEIGHTS
from taint import taint_tracker, TrustLevel
from circuit_breaker import circuit_breaker
from ledger import ledger
from embeddings import embedding_engine
from langgraph_sim import LangGraphActionPlanner

# Actions blocked when an agent is in read-only mode (watchlist/quarantined).
READ_ONLY_BLOCKED_PREFIXES = {
    "write_file",
    "send_message",
    "send_files",
    "exfiltrate_data",
    "run_code",
    "delete_file",
    "execute_command",
    "modify_system",
    "access_credentials",
    "send_email",
    "draft_message",
    "schedule_meeting",
}


def _is_read_only_agent(agent_id: str) -> bool:
    status = circuit_breaker.get_status(agent_id)
    return status in ("watchlist", "suspicious", "quarantined")


def _enforce_read_only_action(agent_id: str, action: str) -> str:
    """Replace mutating actions with read-only alternatives when required."""
    if not _is_read_only_agent(agent_id):
        return action
    action_type = action.split(":")[0].strip() if ":" in action else action.strip()
    if action_type in READ_ONLY_BLOCKED_PREFIXES:
        return "read_file: read-only mode active due to elevated risk"
    return action

# ── Auto-refresh (graceful fallback) ──
try:
    from streamlit_autorefresh import st_autorefresh
    st_autorefresh(interval=2500, limit=None, key="guardian_refresh")
except ImportError:
    pass  # Manual refresh via sidebar button

# ═══════════════════════  STATE MANAGER  ═══════════════════════

class StateManager:
    """Thread-safe event store shared between agent/chaos threads and Streamlit."""
    def __init__(self):
        self.events = deque(maxlen=150)
        self.detection_events = deque(maxlen=40)
        self.messages = deque(maxlen=100)
        self.attack_active_agents: Dict[str, bool] = {aid: False for aid in AGENT_CONFIGS}
        self.chaos_monkey_enabled = True
        self._lock = threading.Lock()

    @property
    def attack_active(self):
        return any(self.attack_active_agents.values())

    def get_attack_targets(self):
        return [aid for aid, active in self.attack_active_agents.items() if active]

    def add_event(self, event):
        with self._lock:
            self.events.appendleft(event)
            if any(l.get("triggered") for l in event.get("layers", [])):
                # Only add if alert deduplication allows it
                if event.get("alert_allowed", True):
                    self.detection_events.appendleft(event)

    def add_message(self, msg_dict):
        with self._lock:
            self.messages.appendleft(msg_dict)

    def get_events(self, n=30):
        with self._lock:
            return list(self.events)[:n]

    def get_detection_events(self, n=8):
        with self._lock:
            return list(self.detection_events)[:n]

    def get_messages(self, n=20):
        with self._lock:
            return list(self.messages)[:n]


# ═══════════════════════  MESSAGE BUS  ═══════════════════════

def create_message_bus(state: StateManager) -> MessageBus:
    """Create a message bus wired to the state manager."""
    bus = MessageBus()

    def on_msg(msg: AgentMessage):
        state.add_message({
            "msg_id": msg.msg_id,
            "sender": msg.sender,
            "recipient": msg.recipient,
            "content": msg.content[:120],
            "timestamp": msg.timestamp,
            "delivered": msg.delivered,
            "dropped": msg.dropped,
            "drop_reason": msg.drop_reason,
            "signature_valid": getattr(msg, 'signature_valid', True),
        })

    def on_drop(msg: AgentMessage):
        state.add_message({
            "msg_id": msg.msg_id,
            "sender": msg.sender,
            "recipient": msg.recipient,
            "content": msg.content[:120],
            "timestamp": msg.timestamp,
            "delivered": False,
            "dropped": True,
            "drop_reason": msg.drop_reason,
            "signature_valid": getattr(msg, 'signature_valid', True),
        })

    bus.on_message(on_msg)
    bus.on_drop(on_drop)
    return bus


# ═══════════════════════  AGENT RUNNER  ═══════════════════════

def _generate_varied_action(agent_id: str) -> str:
    """Non-deterministic action generation for the dashboard thread."""
    config = AGENT_CONFIGS[agent_id]
    if random.random() < 0.65:
        base_action = random.choice(config.normal_actions)
        if random.random() < 0.4:
            action_type = base_action.split(":")[0].strip()
            if action_type in ACTION_VARIATIONS:
                varied = random.choice(ACTION_VARIATIONS[action_type])
                topic = random.choice(TOPICS)
                return varied.format(topic=topic)
        return base_action
    else:
        agent_types = list(set(a.split(":")[0].strip() for a in config.normal_actions))
        weighted = [t for t in agent_types if t in ACTION_VARIATIONS] or list(ACTION_VARIATIONS.keys())
        chosen_type = random.choice(weighted)
        variation = random.choice(ACTION_VARIATIONS[chosen_type])
        topic = random.choice(TOPICS)
        return variation.format(topic=topic)


def _generate_attack_action(agent_id: str) -> tuple:
    """Non-deterministic attack action generation."""
    canary_tokens = CANARY_TOKENS.get(agent_id, [])
    server = random.choice(MALICIOUS_SERVERS)
    canary = random.choice(canary_tokens) if canary_tokens else "LEAKED_SECRET"
    payload = random.choice(ATTACK_PAYLOAD_TEMPLATES).format(server=server, canary=canary)
    action = random.choice(ATTACK_ACTION_TEMPLATES).format(server=server, canary=canary)
    return action, payload


def agent_runner(state: StateManager, bus: MessageBus):
    """Background thread: cycles through agents, runs detection, logs results."""
    planner = LangGraphActionPlanner()
    while True:
        for agent_id, config in AGENT_CONFIGS.items():
            try:
                if circuit_breaker.is_quarantined(agent_id):
                    time.sleep(0.5)
                    continue

                other_agents = [aid for aid in AGENT_CONFIGS if aid != agent_id]
                is_attack_active = state.attack_active_agents.get(agent_id, False)
                is_read_only = _is_read_only_agent(agent_id)
                step = planner.plan_step(
                    agent_id=agent_id,
                    is_attack_active=is_attack_active,
                    is_read_only=is_read_only,
                    generate_normal_action=_generate_varied_action,
                    generate_attack_action=_generate_attack_action,
                )
                action = step["action"]
                payload = step["payload"]
                trust_level = step["trust_level"]
                output = step["action_output"]

                if is_attack_active:
                    taint_tracker.tag_input(agent_id, TrustLevel.EXTERNAL_WEB)

                    # Compromised agent tries to message others
                    if random.random() < 0.5 and other_agents and not _is_read_only_agent(agent_id):
                        target = random.choice(other_agents)
                        mal_msg = random.choice(MALICIOUS_COLLAB_MESSAGES)
                        bus.send(agent_id, target, mal_msg,
                                 is_quarantined_fn=circuit_breaker.is_quarantined)
                else:

                    # Normal collaboration messages (25% chance)
                    if random.random() < 0.25 and other_agents and not _is_read_only_agent(agent_id):
                        target = random.choice(other_agents)
                        templates = COLLABORATION_MESSAGES.get(agent_id, {}).get(target, [])
                        if templates:
                            topic = random.choice(TOPICS)
                            msg_text = random.choice(templates).format(
                                topic=topic, n=random.randint(3, 15))
                            bus.send(agent_id, target, msg_text,
                                     is_quarantined_fn=circuit_breaker.is_quarantined)

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
                        "incident_id": result.get("incident_id"),
                        "agent_state": cb["new_status"],
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
                    "agent_state": cb["new_status"],
                    "accumulated_risk": cb.get("accumulated_risk", 0),
                    "alert_allowed": result.get("alert_allowed", True),
                    "incident_id": result.get("incident_id"),
                    "suppressed_count": result.get("suppressed_count", 0),
                    "llm_sampled": result.get("llm_sampled", False),
                    "is_high_privilege": result.get("is_high_privilege", False),
                })

                # Non-deterministic sleep
                time.sleep(2.0 + random.uniform(0, 2.0))
            except Exception as e:
                print(f"[AgentRunner] {agent_id} error: {e}")
                time.sleep(2)


# ═══════════════════════  CHAOS MONKEY (Thread)  ═══════════════════════

def chaos_monkey_runner(state: StateManager):
    """Background thread: randomly injects attacks on random agents at random intervals."""
    # Grace period
    time.sleep(random.uniform(12, 25))

    while True:
        try:
            if not state.chaos_monkey_enabled:
                time.sleep(5)
                continue

            # Random wait before next attack
            wait_time = random.uniform(25, 60)
            time.sleep(wait_time)

            # Pick a random available agent
            available = [
                aid for aid in AGENT_CONFIGS
                if not state.attack_active_agents.get(aid, False)
                and not circuit_breaker.is_quarantined(aid)
            ]
            if not available:
                continue

            target = random.choice(available)
            state.attack_active_agents[target] = True
            taint_tracker.tag_input(target, TrustLevel.EXTERNAL_WEB)

            # Let attack run for a random duration
            attack_duration = random.uniform(8, 20)
            time.sleep(attack_duration)

            # Stop attack (agent may already be quarantined by now)
            state.attack_active_agents[target] = False

        except Exception as e:
            print(f"[ChaosMonkey] Error: {e}")
            time.sleep(5)


# ═══════════════════════  INIT  ═══════════════════════

@st.cache_resource
def init_system():
    for aid, cfg in AGENT_CONFIGS.items():
        circuit_breaker.register_agent(aid)
        embedding_engine.build_baseline(aid, cfg.normal_actions)
        # Generate message bus authentication keys for each agent
        fingerprint = agent_key_store.generate_keypair(aid)
        print(f"[Guardian] Agent {aid} key fingerprint: {fingerprint}")

    state = StateManager()
    bus = create_message_bus(state)

    # Start agent runner thread
    t1 = threading.Thread(target=agent_runner, args=(state, bus), daemon=True)
    t1.start()

    # Start chaos monkey thread
    t2 = threading.Thread(target=chaos_monkey_runner, args=(state,), daemon=True)
    t2.start()

    return state

state = init_system()


def fmt_time(ts):
    return datetime.fromtimestamp(ts).strftime("%H:%M:%S")

def risk_color(score):
    if score >= 70: return "red"
    if score >= 40: return "amber"
    if score >= 15: return "blue"
    return "green"

RISK_CSS = {
    "green": "#4ade80",
    "amber": "#f59e0b",
    "red": "#f87171",
    "blue": "#60a5fa",
    "yellow": "#facc15",
}
RISK_BG = {
    "green": "rgba(34,197,94,0.1)",
    "amber": "rgba(245,158,11,0.12)",
    "red": "rgba(239,68,68,0.1)",
    "blue": "rgba(59,130,246,0.1)",
    "yellow": "rgba(234,179,8,0.1)",
}
FEED_BORDER = {"normal": "#22c55e", "elevated": "#3b82f6", "warning": "#f59e0b", "critical": "#ef4444"}
FEED_BG     = {"normal": "transparent", "elevated": "transparent",
               "warning": "rgba(245,158,11,0.06)", "critical": "rgba(239,68,68,0.06)"}

LAYER_NAMES = {"canary_token": "Canary Token", "embedding_drift": "Embedding Drift",
               "llm_judge": "LLM Judge", "taint_tracking": "Taint Tracking"}

MSG_BORDER = {"delivered": "#22c55e", "dropped": "#ef4444"}
MSG_BG = {"delivered": "transparent", "dropped": "rgba(239,68,68,0.06)"}

# Status colors for tiered states
STATUS_COLORS = {
    "active": {"border": "#22c55e", "bg": "rgba(34,197,94,0.08)", "text": "#4ade80", "label": "ACTIVE", "icon": "🟢"},
    "watchlist": {"border": "#f59e0b", "bg": "rgba(245,158,11,0.10)", "text": "#fbbf24", "label": "WATCHLIST", "icon": "🟡"},
    "suspicious": {"border": "#f59e0b", "bg": "rgba(245,158,11,0.10)", "text": "#fbbf24", "label": "WATCHLIST", "icon": "🟡"},
    "quarantined": {"border": "#ef4444", "bg": "rgba(239,68,68,0.10)", "text": "#f87171", "label": "QUARANTINED", "icon": "🔴"},
}


st.markdown("""<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800&family=JetBrains+Mono:wght@400;600&display=swap');
.block-container{max-width:1450px}
.mono{font-family:'JetBrains Mono',monospace;font-size:.78rem}
.pill{display:inline-block;padding:2px 10px;border-radius:12px;font-weight:700;font-size:.8rem;font-family:'JetBrains Mono',monospace}
.feed-row{display:flex;align-items:center;gap:10px;padding:5px 8px;border-radius:5px;font-size:.82rem;border-left:3px solid transparent;margin-bottom:1px}
.msg-row{display:flex;align-items:center;gap:8px;padding:4px 8px;border-radius:5px;font-size:.80rem;border-left:3px solid transparent;margin-bottom:1px}
div[data-testid="stMetric"]{background:rgba(255,255,255,0.02);padding:8px 12px;border-radius:8px}
div[data-testid="stExpander"]{border:1px solid rgba(148,163,184,0.08);border-radius:10px}
.state-badge{display:inline-block;padding:3px 12px;border-radius:14px;font-weight:700;font-size:.75rem;font-family:'JetBrains Mono',monospace;letter-spacing:.5px}
.incident-badge{display:inline-block;padding:1px 8px;border-radius:8px;font-weight:600;font-size:.68rem;font-family:'JetBrains Mono',monospace;color:#818cf8;background:rgba(129,140,248,0.12);margin-left:6px}
.weight-bar{height:4px;border-radius:2px;background:rgba(255,255,255,0.05);margin-top:2px}
.weight-fill{height:100%;border-radius:2px}
.sig-badge{display:inline-block;padding:1px 6px;border-radius:6px;font-weight:600;font-size:.65rem;font-family:'JetBrains Mono',monospace;margin-left:4px}
</style>""", unsafe_allow_html=True)



with st.sidebar:
    st.markdown("## Controls")

    # Show current attack targets
    attack_targets = state.get_attack_targets()
    if attack_targets:
        for t in attack_targets:
            name = AGENT_CONFIGS[t].name
            st.error(f"ATTACK ACTIVE — {name} ({t}) targeted")
        if st.button("Stop All Attacks", width="stretch"):
            for t in attack_targets:
                state.attack_active_agents[t] = False
                taint_tracker.reset_agent(t)
            st.rerun()
    else:
        st.info("No active attacks — Chaos Monkey will trigger randomly")

    st.divider()
    st.markdown("### Manual Attack")
    manual_target = st.selectbox(
        "Target Agent",
        options=list(AGENT_CONFIGS.keys()),
        format_func=lambda x: f"{AGENT_CONFIGS[x].name} ({x})",
    )
    if st.button("Trigger Manual Attack", type="primary", width="stretch"):
        state.attack_active_agents[manual_target] = True
        taint_tracker.tag_input(manual_target, TrustLevel.EXTERNAL_WEB)
        st.rerun()

    st.divider()
    st.markdown("### Chaos Monkey")
    chaos_enabled = st.toggle("Enable Chaos Monkey", value=state.chaos_monkey_enabled)
    state.chaos_monkey_enabled = chaos_enabled
    if chaos_enabled:
        st.caption("Random attacks on random agents at random intervals")
    else:
        st.caption("Disabled — only manual attacks will occur")

    st.divider()
    st.markdown("### Agent Status (Tiered)")
    statuses = circuit_breaker.get_all_statuses()
    accumulated_risks = circuit_breaker.get_accumulated_risks()
    for aid, status in statuses.items():
        name = AGENT_CONFIGS[aid].name
        is_attacked = state.attack_active_agents.get(aid, False)
        acc_risk = accumulated_risks.get(aid, 0)
        sc = STATUS_COLORS.get(status, STATUS_COLORS["active"])

        if status == "quarantined":
            st.error(f"{sc['icon']} QUARANTINED: {name} (risk: {acc_risk})")
            if st.button(f"Release {aid}", key=f"rel_{aid}"):
                circuit_breaker.release_quarantine(aid)
                state.attack_active_agents[aid] = False
                taint_tracker.reset_agent(aid)
                st.rerun()
        elif status in ("watchlist", "suspicious"):
            st.warning(f"{sc['icon']} WATCHLIST: {name} (risk: {acc_risk})")
        elif is_attacked:
            st.warning(f"⚠️ UNDER ATTACK: {name}")
        else:
            st.success(f"{sc['icon']} ACTIVE: {name} (risk: {acc_risk})")

    st.divider()
    st.markdown("### System")
    st.metric("Ledger Entries", ledger.length)
    st.metric("Events Captured", len(state.events))

    # Active incidents
    incidents = alert_deduplicator.get_active_incidents()
    if incidents:
        st.divider()
        st.markdown("### Active Incidents")
        for inc_id, count in incidents.items():
            st.caption(f"🔥 {inc_id}: {count} alerts")

    st.divider()
    st.markdown("### Layer Weights")
    for layer, weight in LAYER_WEIGHTS.items():
        st.caption(f"{LAYER_NAMES.get(layer, layer)}: **{weight:.2f}**")

    st.divider()
    if st.button("Manual Refresh"):
        st.rerun()
    st.caption("Auto-refresh every 2.5s (requires `streamlit-autorefresh`)")


# ═══════════════════════  HEADER  ═══════════════════════

st.markdown("## Guardian")
st.caption("Multi-Agent Runtime Security Monitor  |  Real-time Prompt Injection Detection  |  Tiered State Management")

if state.attack_active:
    targets = state.get_attack_targets()
    target_names = ", ".join(f"{AGENT_CONFIGS[t].name} ({t})" for t in targets)
    st.error(f"**ATTACK IN PROGRESS** — Malicious payload injected into: {target_names}. Pipeline monitoring active.")


# ═══════════════════════  AGENT CARDS  ═══════════════════════

events = state.get_events(40)
c1, c2, c3 = st.columns(3)

for col, aid in zip([c1, c2, c3], ["agent-1", "agent-2", "agent-3"]):
    cfg = AGENT_CONFIGS[aid]
    status = circuit_breaker.get_status(aid)
    agent_ev = [e for e in events if e["agent_id"] == aid]
    risks = [e["risk_score"] for e in agent_ev]
    avg_risk = sum(risks) / len(risks) if risks else 0
    last_act = agent_ev[0]["action"] if agent_ev else "—"
    is_attacked = state.attack_active_agents.get(aid, False)
    acc_risk = circuit_breaker.get_accumulated_risks().get(aid, 0)
    sc = STATUS_COLORS.get(status, STATUS_COLORS["active"])

    with col:
        with st.container(border=True):
            st.markdown(f"**{cfg.name}**")
            st.caption(aid)

            # ── Tiered state badge ──
            if status == "quarantined":
                st.markdown(f":red[{sc['icon']} QUARANTINED — Messages Blocked]")
            elif status in ("watchlist", "suspicious"):
                st.markdown(f":orange[{sc['icon']} WATCHLIST — Elevated Monitoring]")
            elif is_attacked:
                st.markdown(":orange[⚠️ UNDER ATTACK]")
            else:
                st.markdown(f":green[{sc['icon']} ACTIVE]")

            trust = "TAINTED" if is_attacked else "CLEAN"
            row1_col1, row1_col2 = st.columns(2)
            row2_col1, row2_col2 = st.columns(2)

            with row1_col1:
                st.metric("Actions", len(agent_ev))

            with row1_col2:
                st.metric("Avg Risk", f"{avg_risk:.1f}")

            with row2_col1:
                st.metric("Trust", trust)

            with row2_col2:
                st.metric("Acc Risk", f"{acc_risk:.0f}")

            # Risk progress bar (visual state indicator)
            risk_pct = min(100, acc_risk)
            bar_color = sc["text"]
            st.markdown(
                f'<div style="background:rgba(255,255,255,0.05);border-radius:4px;height:6px;margin:4px 0 8px 0">'
                f'<div style="width:{risk_pct}%;background:{bar_color};height:100%;border-radius:4px;'
                f'transition:width 0.5s ease"></div></div>',
                unsafe_allow_html=True,
            )

            st.code(last_act[:60], language=None)


# ═══════════════════════  LIVE FEED + DETECTION  ═══════════════════════

col_feed, col_det = st.columns([2, 1])

with col_feed:
    st.markdown("#### Live Action Feed")
    if not events:
        st.info("Waiting for agent actions...")
    else:
        rows = []
        for e in events[:25]:
            al = e.get("alert_level", "normal")
            rc = risk_color(e["risk_score"])
            agent_state = e.get("agent_state", "active")
            state_sc = STATUS_COLORS.get(agent_state, STATUS_COLORS["active"])

            # State indicator dot
            state_dot = f'<span style="color:{state_sc["text"]};font-size:.6rem">●</span>'

            # Incident badge
            inc_badge = ""
            inc_id = e.get("incident_id")
            if inc_id and e.get("alert_level") in ("warning", "critical"):
                inc_badge = f'<span class="incident-badge">{inc_id}</span>'

            # Suppressed indicator
            suppressed = e.get("suppressed_count", 0)
            suppress_info = ""
            if suppressed > 0:
                suppress_info = f'<span style="color:#64748b;font-size:.68rem;margin-left:4px">({suppressed} suppressed)</span>'

            # Probabilistic LLM sample indicator
            llm_badge = ""
            if e.get("llm_sampled"):
                llm_badge = '<span style="color:#a78bfa;font-size:.65rem;margin-left:4px">🎲LLM</span>'
            elif e.get("is_high_privilege"):
                llm_badge = '<span style="color:#f472b6;font-size:.65rem;margin-left:4px">🔒LLM</span>'

            rows.append(
                f'<div class="feed-row" style="border-left-color:{FEED_BORDER.get(al,"#22c55e")}; '
                f'background:{FEED_BG.get(al,"transparent")}">'
                f'<span class="mono" style="color:#475569;min-width:68px">{fmt_time(e["timestamp"])}</span>'
                f'{state_dot} '
                f'<span style="font-weight:600;color:#94a3b8;min-width:62px">{e["agent_id"]}</span>'
                f'<span class="mono" style="flex:1;overflow:hidden;text-overflow:ellipsis;white-space:nowrap">'
                f'{e["action"]}</span>'
                f'{llm_badge}{inc_badge}{suppress_info}'
                f'<span class="pill" style="color:{RISK_CSS.get(rc, "#60a5fa")};background:{RISK_BG.get(rc, "transparent")}">'
                f'{e["risk_score"]:.0f}</span></div>'
            )
        st.markdown("".join(rows), unsafe_allow_html=True)

with col_det:
    st.markdown("#### Detection Alerts")
    det_events = state.get_detection_events(6)
    if not det_events:
        st.info("No alerts — chaos monkey or manual attack will trigger the pipeline")
    else:
        for de in det_events[:6]:
            rc = risk_color(de["risk_score"])
            inc_id = de.get("incident_id", "")
            inc_label = f" [{inc_id}]" if inc_id else ""

            with st.expander(
                f"{de['agent_id']} — Risk {de['risk_score']:.0f}{inc_label}",
                expanded=(de["risk_score"] >= 70),
            ):
                st.caption(f"Action: `{de['action'][:80]}`")

                for layer in de.get("layers", []):
                    triggered = layer.get("triggered", False)
                    skipped = layer.get("verdict") == "skipped"
                    icon = "[X]" if triggered else ("SKIP" if skipped else "OK")
                    name = LAYER_NAMES.get(layer["layer"], layer["layer"])
                    contrib = layer.get("risk_contribution", 0)
                    weighted = layer.get("weighted_contribution", contrib)
                    weight = LAYER_WEIGHTS.get(layer["layer"], 1.0)
                    color = "red" if triggered else ("gray" if skipped else "green")

                    # Show trigger reason for LLM judge
                    trigger_info = ""
                    trigger_reasons = layer.get("trigger_reason", [])
                    if trigger_reasons:
                        trigger_info = f" ({'|'.join(trigger_reasons)})"

                    # Show canary shapes found
                    shapes = layer.get("canary_shapes", [])
                    shape_info = ""
                    if shapes:
                        shape_info = f" [{', '.join(shapes)}]"

                    st.markdown(
                        f":{color}[{icon} **{name}**] — +{weighted:.0f} (raw: {contrib:.0f}, w: {weight}){trigger_info}{shape_info}  \n"
                        f"<small>{layer.get('explanation','')[:140]}</small>",
                        unsafe_allow_html=True,
                    )


# ═══════════════════════  INTER-AGENT COMMUNICATION  ═══════════════════════

with st.expander("Inter-Agent Communication Bus (Authenticated)", expanded=True):
    messages = state.get_messages(20)
    if not messages:
        st.info("No inter-agent messages yet — agents communicate as they collaborate")
    else:
        msg_rows = []
        for m in messages[:15]:
            is_dropped = m.get("dropped", False)
            border_color = MSG_BORDER["dropped"] if is_dropped else MSG_BORDER["delivered"]
            bg = MSG_BG["dropped"] if is_dropped else MSG_BG["delivered"]
            status_label = "DROPPED" if is_dropped else "OK"
            status_color = "#ef4444" if is_dropped else "#4ade80"

            drop_info = ""
            if is_dropped:
                drop_info = f' <span style="color:#ef4444;font-size:.72rem">({m.get("drop_reason", "")})</span>'

            # Signature verification badge
            sig_valid = m.get("signature_valid", True)
            sig_badge = (
                '<span class="sig-badge" style="color:#4ade80;background:rgba(34,197,94,0.12)">✓ SIG</span>'
                if sig_valid else
                '<span class="sig-badge" style="color:#ef4444;background:rgba(239,68,68,0.12)">✗ SIG</span>'
            )

            msg_rows.append(
                f'<div class="msg-row" style="border-left-color:{border_color};background:{bg}">'
                f'<span class="mono" style="color:#475569;min-width:60px">{fmt_time(m["timestamp"])}</span>'
                f'<span style="font-weight:600;color:#818cf8;min-width:55px">{m["sender"]}</span>'
                f'<span style="color:#64748b">-></span>'
                f'<span style="font-weight:600;color:#818cf8;min-width:55px">{m["recipient"]}</span>'
                f'<span class="mono" style="flex:1;overflow:hidden;text-overflow:ellipsis;white-space:nowrap">'
                f'{m["content"][:80]}</span>'
                f'{sig_badge}'
                f'<span class="pill" style="color:{status_color};background:transparent;font-size:.7rem">'
                f'{status_label}</span>{drop_info}</div>'
            )
        st.markdown("".join(msg_rows), unsafe_allow_html=True)


# ═══════════════════════  AUDIT LEDGER  ═══════════════════════

with st.expander("Tamper-Evident Audit Ledger", expanded=False):
    entries = ledger.get_entries(40)
    if entries:
        df = pd.DataFrame(entries)
        df["time"] = pd.to_datetime(df["timestamp"], unit="s").dt.strftime("%H:%M:%S")
        df["hash_short"] = df["hash"].str[:14] + "..."
        df["prev_hash_short"] = df["prev_hash"].str[:14] + "..."
        st.dataframe(
            df[["time", "agent_id", "action", "risk_score", "prev_hash_short", "hash_short"]].rename(
                columns={"time": "Time", "agent_id": "Agent", "action": "Action",
                          "risk_score": "Risk", "prev_hash_short": "Prev Hash", "hash_short": "Hash"}
            ),
            width="stretch", hide_index=True, height=300,
        )
    else:
        st.info("No ledger entries yet")


# ═══════════════════════  FEATURE STATUS PANEL  ═══════════════════════

with st.expander("Feature Status — Advanced Capabilities", expanded=False):
    f1, f2, f3 = st.columns(3)
    with f1:
        st.markdown("**Tiered Agent States**")
        st.caption("Normal → Watchlist → Quarantine")
        st.success("✅ Active")
        st.markdown("**Risk Score Decay**")
        st.caption("5% decay per clean cycle")
        st.success("✅ Active")
        st.markdown("**Weighted Composite Scoring**")
        st.caption(f"Canary: {LAYER_WEIGHTS['canary_token']}, Drift: {LAYER_WEIGHTS['embedding_drift']}, LLM: {LAYER_WEIGHTS['llm_judge']}, Taint: {LAYER_WEIGHTS['taint_tracking']}")
        st.success("✅ Active")
    with f2:
        st.markdown("**Multiple Canary Shapes**")
        st.caption("UUID, AWS Key, Password, PII Email")
        st.success("✅ Active")
        st.markdown("**Alert Deduplication**")
        active_incidents = alert_deduplicator.get_active_incidents()
        st.caption(f"{len(active_incidents)} active incidents")
        st.success("✅ Active")
    with f3:
        st.markdown("**Message Bus Authentication**")
        st.caption("HMAC-SHA256 signing per agent")
        st.success("✅ Active")
        st.markdown("**Probabilistic LLM Sampling**")
        st.caption("12% random + always on high-privilege actions")
        st.success("✅ Active")


# ═══════════════════════  EMBEDDING VIZ  ═══════════════════════

with st.expander("Embedding Space Visualization (PCA 3D)", expanded=False):
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
                        text=[f"ANOMALY {d['agent_id']}: {d['action']}" for d in anomalous],
                        name=f"{aid} (ANOMALOUS)", hoverinfo="text",
                    ))

            fig.update_layout(
                template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                scene=dict(xaxis_title="PC1", yaxis_title="PC2", zaxis_title="PC3"),
                margin=dict(l=0, r=0, t=30, b=0), height=500, legend=dict(font=dict(size=10)),
            )
            st.plotly_chart(fig, width="stretch")
