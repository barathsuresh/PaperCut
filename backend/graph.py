from langgraph.graph import END, StateGraph

from backend.nodes.node0_validator import run_node0
from backend.nodes.node1_ingestor import run_node1
from backend.nodes.node2_client import run_node2
from backend.nodes.node3_client import run_node3
from backend.schemas.state import AgentState


async def _node0(state: AgentState) -> dict:
    try:
        result = await run_node0(state["pdf_gcs_uri"])
        return {
            "scope_valid": result.result == "PASS",
            "scope_reason": result.reason,
        }
    except Exception as e:
        return {"scope_valid": False, "scope_reason": str(e), "error": str(e)}


async def _node1(state: AgentState) -> dict:
    try:
        blueprint = await run_node1(state["pdf_gcs_uri"])
        return {"blueprint": blueprint}
    except Exception as e:
        return {"error": str(e)}


async def _node2(state: AgentState) -> dict:
    blueprint = state.get("blueprint")
    result = run_node2(blueprint.model_dump() if blueprint else {})
    return {"scaffold_code": result.get("status")}


async def _node3(state: AgentState) -> dict:
    blueprint = state.get("blueprint")
    result = run_node3(blueprint.model_dump() if blueprint else {})
    return {"cuda_blueprint": result}


def _route_after_node0(state: AgentState) -> str:
    return "node1" if state.get("scope_valid") else END


def build_graph():
    graph = StateGraph(AgentState)

    graph.add_node("node0", _node0)
    graph.add_node("node1", _node1)
    graph.add_node("node2", _node2)
    graph.add_node("node3", _node3)

    graph.set_entry_point("node0")
    graph.add_conditional_edges("node0", _route_after_node0)
    graph.add_edge("node1", "node2")
    graph.add_edge("node2", "node3")
    graph.add_edge("node3", END)

    return graph.compile()


pipeline = build_graph()
