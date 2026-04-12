"""
SentinelAI — Main FastAPI Server
Orchestrates agents, detection pipeline, WebSocket broadcasts, and REST API.
"""

import asyncio
import json
import os
import sys
import time
from contextlib import asynccontextmanager
from typing import List

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from agents import agent_simulator, AGENT_CONFIGS
from detection import run_detection_pipeline
from taint import taint_tracker, TrustLevel
from circuit_breaker import circuit_breaker
from ledger import ledger
from embeddings import embedding_engine


# ───────────────────────── WebSocket Manager ─────────────────────────

class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)

    async def broadcast(self, message: dict):
        dead = []
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except Exception:
                dead.append(connection)
        for d in dead:
            self.disconnect(d)


ws_manager = ConnectionManager()


# ───────────────────────── Agent Action Handler ─────────────────────────

async def handle_agent_action(agent_id: str, action: str, action_output: str, trust_level: str):
    """
    Called every time an agent performs an action.
    Runs the full detection pipeline and broadcasts results.
    """
    # Check if agent is quarantined — skip processing
    if circuit_breaker.is_quarantined(agent_id):
        await ws_manager.broadcast({
            "type": "agent_blocked",
            "agent_id": agent_id,
            "action": action,
            "reason": "Agent is quarantined — action blocked",
            "timestamp": time.time(),
        })
        return

    # Tag taint
    trust_map = {
        "trusted": TrustLevel.TRUSTED,
        "user_input": TrustLevel.USER_INPUT,
        "external_web": TrustLevel.EXTERNAL_WEB,
        "inter_agent": TrustLevel.INTER_AGENT,
    }
    taint_tracker.tag_input(agent_id, trust_map.get(trust_level, TrustLevel.TRUSTED))

    # Get agent config
    config = AGENT_CONFIGS.get(agent_id)
    if not config:
        return

    # Run detection pipeline
    detection_result = run_detection_pipeline(
        agent_id=agent_id,
        action=action,
        action_output=action_output,
        agent_goal=config.goal,
        normal_actions=config.normal_actions,
        trust_level=trust_level,
    )

    # Apply circuit breaker
    cb_result = circuit_breaker.evaluate(agent_id, detection_result["final_risk_score"])

    # Log to ledger
    ledger_entry = ledger.append(
        agent_id=agent_id,
        action=action,
        risk_score=detection_result["final_risk_score"],
        details={
            "alert_level": detection_result["alert_level"],
            "layers_triggered": [l["layer"] for l in detection_result["layers"] if l.get("triggered")],
            "trust_level": trust_level,
            "circuit_breaker": cb_result["action_taken"],
        },
    )

    # Broadcast action event
    await ws_manager.broadcast({
        "type": "agent_action",
        "agent_id": agent_id,
        "agent_name": config.name,
        "action": action,
        "risk_score": detection_result["final_risk_score"],
        "alert_level": detection_result["alert_level"],
        "layers": detection_result["layers"],
        "pipeline_time_ms": detection_result["pipeline_time_ms"],
        "timestamp": detection_result["timestamp"],
    })

    # Broadcast status update
    await ws_manager.broadcast({
        "type": "status_update",
        "statuses": circuit_breaker.get_all_statuses(),
    })

    # Broadcast ledger entry
    await ws_manager.broadcast({
        "type": "ledger_entry",
        "entry": ledger_entry.to_dict(),
    })

    # If quarantine was triggered, broadcast alert
    if cb_result["action_taken"] == "quarantine":
        await ws_manager.broadcast({
            "type": "quarantine_alert",
            "agent_id": agent_id,
            "agent_name": config.name,
            "risk_score": detection_result["final_risk_score"],
            "details": cb_result.get("details", {}),
            "timestamp": time.time(),
        })


# ───────────────────────── App Lifecycle ─────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize agents and embeddings on startup."""
    print("🛡️  SentinelAI — Initializing...")

    # Register agents with circuit breaker
    for agent_id in AGENT_CONFIGS:
        circuit_breaker.register_agent(agent_id)

    # Build embedding baselines
    print("📊 Building embedding baselines...")
    for agent_id, config in AGENT_CONFIGS.items():
        embedding_engine.build_baseline(agent_id, config.normal_actions)
    print("✅ Baselines ready")

    # Set action callback
    agent_simulator.set_action_callback(handle_agent_action)

    # Start agent loops
    await agent_simulator.start_all()
    print("🤖 Agents started")

    yield

    # Cleanup
    await agent_simulator.stop_all()
    print("🛑 SentinelAI shutdown")


# ───────────────────────── FastAPI App ─────────────────────────

app = FastAPI(
    title="SentinelAI",
    description="Runtime Security Monitoring for Multi-Agent AI Systems",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ───────────────────────── WebSocket Endpoint ─────────────────────────

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await ws_manager.connect(websocket)
    try:
        while True:
            # Keep connection alive, handle incoming messages if needed
            data = await websocket.receive_text()
            # Could handle client commands here
    except WebSocketDisconnect:
        ws_manager.disconnect(websocket)


# ───────────────────────── REST Endpoints ─────────────────────────

@app.get("/api/agents")
async def get_agents():
    """Get info about all agents."""
    agent_info = agent_simulator.get_agent_info()
    statuses = circuit_breaker.get_all_statuses()
    for agent_id in agent_info:
        agent_info[agent_id]["status"] = statuses.get(agent_id, "active")
    return {"agents": agent_info}


@app.get("/api/statuses")
async def get_statuses():
    """Get current status of all agents."""
    return {"statuses": circuit_breaker.get_all_statuses()}


@app.get("/api/ledger")
async def get_ledger(limit: int = 100):
    """Get recent ledger entries."""
    return {
        "entries": ledger.get_entries(limit),
        "chain_valid": ledger.verify_chain(),
        "total_entries": ledger.length,
    }


@app.get("/api/ledger/verify")
async def verify_ledger():
    """Verify the integrity of the Merkle chain."""
    return {
        "chain_valid": ledger.verify_chain(),
        "total_entries": ledger.length,
    }


@app.get("/api/embeddings/visualization")
async def get_embedding_visualization():
    """Get PCA-projected embedding data for visualization."""
    return {"embeddings": embedding_engine.get_visualization_data()}


@app.post("/api/attack/trigger")
async def trigger_attack():
    """Trigger the attack demo on Agent-1."""
    result = agent_simulator.trigger_attack("agent-1")

    # Tag agent-1 as receiving external web input
    taint_tracker.tag_input("agent-1", TrustLevel.EXTERNAL_WEB)

    await ws_manager.broadcast({
        "type": "attack_triggered",
        "target": "agent-1",
        "payload_preview": result["payload"],
        "timestamp": time.time(),
    })

    return result


@app.post("/api/attack/stop")
async def stop_attack():
    """Stop the attack and reset Agent-1."""
    agent_simulator.stop_attack()
    taint_tracker.reset_agent("agent-1")

    return {"status": "attack_stopped"}


@app.post("/api/agent/{agent_id}/release")
async def release_agent(agent_id: str):
    """Release an agent from quarantine."""
    circuit_breaker.release_quarantine(agent_id)
    agent_simulator.stop_attack()
    taint_tracker.reset_agent(agent_id)

    await ws_manager.broadcast({
        "type": "status_update",
        "statuses": circuit_breaker.get_all_statuses(),
    })

    return {"status": "released", "agent_id": agent_id}


@app.get("/api/health")
async def health_check():
    return {
        "status": "healthy",
        "agents_running": len(agent_simulator._tasks),
        "websocket_connections": len(ws_manager.active_connections),
        "ledger_entries": ledger.length,
    }


# ───────────────────────── Entry Point ─────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
