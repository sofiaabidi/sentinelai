

from typing import Callable, Dict, TypedDict

try:
    from langgraph.graph import END, START, StateGraph
    LANGGRAPH_AVAILABLE = True
except Exception:
    LANGGRAPH_AVAILABLE = False


class SimulationState(TypedDict, total=False):
    agent_id: str
    is_attack_active: bool
    is_read_only: bool
    action: str
    payload: str
    trust_level: str
    action_output: str
    generated_attack_action: str
    generated_attack_payload: str
    generated_normal_action: str


class LangGraphActionPlanner:
  

    def __init__(self):
        self._graph = self._build_graph() if LANGGRAPH_AVAILABLE else None

    def _build_graph(self):
        graph = StateGraph(SimulationState)
        graph.add_node("select_mode", self._select_mode)
        graph.add_node("apply_read_only", self._apply_read_only)
        graph.add_node("finalize_output", self._finalize_output)
        graph.add_edge(START, "select_mode")
        graph.add_edge("select_mode", "apply_read_only")
        graph.add_edge("apply_read_only", "finalize_output")
        graph.add_edge("finalize_output", END)
        return graph.compile()

    @staticmethod
    def _select_mode(state: SimulationState) -> SimulationState:
        if state["is_attack_active"]:
            action = state.get("generated_attack_action", "read_file: no-op")
            payload = state.get("generated_attack_payload", "")
            trust_level = "external_web"
            return {"action": action, "payload": payload, "trust_level": trust_level}
        action = state.get("generated_normal_action", "read_file: no-op")
        return {"action": action, "payload": "", "trust_level": "trusted"}

    @staticmethod
    def _apply_read_only(state: SimulationState) -> SimulationState:
        action = state.get("action", "read_file: no-op")
        action_type = action.split(":")[0].strip() if ":" in action else action.strip()
        if state.get("is_read_only", False):
            blocked = {
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
            if action_type in blocked:
                action = "read_file: read-only mode active due to elevated risk"
        return {"action": action}

    @staticmethod
    def _finalize_output(state: SimulationState) -> SimulationState:
        action = state.get("action", "read_file: no-op")
        payload = state.get("payload", "")
        if state.get("is_attack_active", False):
            output = (payload + " " + action).strip()
        else:
            output = f"Completed: {action}"
        return {"action_output": output}

    def plan_step(
        self,
        agent_id: str,
        is_attack_active: bool,
        is_read_only: bool,
        generate_normal_action: Callable[[str], str],
        generate_attack_action: Callable[[str], tuple],
    ) -> Dict:
        attack_action, attack_payload = generate_attack_action(agent_id)
        normal_action = generate_normal_action(agent_id)

        seed_state: SimulationState = {
            "agent_id": agent_id,
            "is_attack_active": is_attack_active,
            "is_read_only": is_read_only,
            "generated_attack_action": attack_action,
            "generated_attack_payload": attack_payload,
            "generated_normal_action": normal_action,
        }

        if self._graph is not None:
            result = self._graph.invoke(seed_state)
        else:
            # Safe fallback: equivalent control flow if langgraph isn't importable.
            result = self._finalize_output(
                {**seed_state, **self._apply_read_only({**seed_state, **self._select_mode(seed_state)})}
            )
            result = {**seed_state, **self._select_mode(seed_state), **self._apply_read_only({**seed_state, **self._select_mode(seed_state)}), **result}

        return {
            "action": result.get("action", "read_file: no-op"),
            "payload": result.get("payload", ""),
            "trust_level": result.get("trust_level", "trusted"),
            "action_output": result.get("action_output", "Completed: read_file: no-op"),
        }

