import asyncio
import logging

from langgraph.graph import END, StateGraph

from backend.nodes.node0_validator import run_node0
from backend.nodes.node1_ingestor import run_node1
from backend.nodes.node2_client import run_node2
from backend.nodes.node3_client import run_node3
from backend.schemas.state import AgentState

logger = logging.getLogger(__name__)


async def _node0(state: AgentState) -> dict:
    logger.info("Graph — entering node0 | uri=%s", state.get("pdf_gcs_uri"))
    try:
        result = await run_node0(state["pdf_gcs_uri"])
        scope_valid = result.result == "PASS"
        logger.info("Graph — node0 done | scope_valid=%s", scope_valid)
        return {
            "scope_valid": scope_valid,
            "scope_reason": result.reason,
        }
    except Exception as e:
        logger.error("Graph — node0 error: %s", e)
        return {"scope_valid": False, "scope_reason": str(e), "error": str(e)}


async def _node1(state: AgentState) -> dict:
    from backend.nodes.node1_ingestor import ScopeRejectedError
    logger.info("Graph — entering node1")
    try:
        blueprint = await run_node1(state["pdf_gcs_uri"])
        logger.info("Graph — node1 done | model_type=%s", blueprint.model_type)
        return {"blueprint": blueprint}
    except ScopeRejectedError as e:
        logger.warning("Graph — node1 scope rejected: %s", e.reason)
        return {"scope_valid": False, "scope_reason": e.reason}
    except Exception as e:
        logger.error("Graph — node1 error: %s", e)
        return {"error": str(e)}


async def _node2(state: AgentState) -> dict:
    logger.info("Graph — entering node2")
    blueprint = state.get("blueprint")
    result = await asyncio.to_thread(
        run_node2,
        blueprint.model_dump() if blueprint else {},
        state.get("session_id"),
    )
    logger.info("Graph — node2 done | status=%s", result.get("status"))
    return {"scaffold_code": result}


async def _node3(state: AgentState) -> dict:
    logger.info("Graph — entering node3")
    # Pass scaffold_code (node2 result) so node3 can locate the correct output_dir
    node2_result = state.get("scaffold_code") or {}
    result = await asyncio.to_thread(run_node3, node2_result, state.get("session_id"))
    logger.info("Graph — node3 done | status=%s", result.get("status"))
    return {"cuda_blueprint": result}


def _route_after_node0(state: AgentState) -> str:
    route = "node1" if state.get("scope_valid") else END
    logger.info("Graph — routing after node0 → %s", route)
    return route


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
