"""
End-to-end integration test: Node 2 → Node 3 via LangGraph.

Bypasses Nodes 0 and 1 (they require live GCP). Injects the sample
contract directly into the graph state and runs Node 2 + Node 3 against
the real NVIDIA API.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from langgraph.graph import END, StateGraph

from backend.nodes.node2_client import run_node2
from backend.nodes.node3_client import run_node3
from backend.schemas.contract import ArchitectureBlueprint
from backend.schemas.state import AgentState

SAMPLE_CONTRACT = (
    Path(__file__).resolve().parent.parent.parent
    / "contracts" / "sample_attention_is_all_you_need.json"
)

SCAFFOLD_DIR = (
    Path(__file__).resolve().parent.parent.parent
    / "outputs" / "pytorch_scaffold"
)

BLUEPRINT_DIR = (
    Path(__file__).resolve().parent.parent.parent
    / "outputs" / "hardware_blueprint"
)


def _build_node2_node3_graph():
    """Build a minimal sub-graph: node2 → node3 → END."""

    async def _node2(state: AgentState) -> dict:
        blueprint = state.get("blueprint")
        result = run_node2(blueprint.model_dump() if blueprint else {})
        return {"scaffold_code": result.get("status")}

    async def _node3(state: AgentState) -> dict:
        blueprint = state.get("blueprint")
        result = run_node3(blueprint.model_dump() if blueprint else {})
        return {"cuda_blueprint": result}

    graph = StateGraph(AgentState)
    graph.add_node("node2", _node2)
    graph.add_node("node3", _node3)
    graph.set_entry_point("node2")
    graph.add_edge("node2", "node3")
    graph.add_edge("node3", END)
    return graph.compile()


@pytest.mark.integration
class TestNode2Node3Integration:
    """Runs Node 2 and Node 3 end-to-end against the real NVIDIA API."""

    @pytest.fixture(autouse=True)
    def load_blueprint(self):
        """Load the sample contract and parse it as an ArchitectureBlueprint."""
        raw = json.loads(SAMPLE_CONTRACT.read_text())
        self.blueprint = ArchitectureBlueprint(**raw)

    @pytest.mark.asyncio
    async def test_full_pipeline_node2_through_node3(self):
        graph = _build_node2_node3_graph()

        initial_state: AgentState = {
            "session_id": "integration-test",
            "pdf_gcs_uri": "",
            "scope_valid": True,
            "scope_reason": "bypassed",
            "blueprint": self.blueprint,
        }

        final_state = await graph.ainvoke(initial_state)

        # --- Node 2 assertions ---
        assert final_state.get("scaffold_code") == "completed", (
            f"Node 2 did not complete: scaffold_code={final_state.get('scaffold_code')}"
        )

        model_py = SCAFFOLD_DIR / "model.py"
        assert model_py.exists(), "outputs/pytorch_scaffold/model.py not found"
        model_content = model_py.read_text()
        assert "class" in model_content, "model.py should define at least one class"
        assert "nn.Module" in model_content or "Module" in model_content, (
            "model.py should contain nn.Module"
        )

        # --- Node 3 assertions ---
        cuda_result = final_state.get("cuda_blueprint", {})
        assert cuda_result.get("status") == "completed", (
            f"Node 3 did not complete: {cuda_result}"
        )

        cu_files = list(BLUEPRINT_DIR.glob("*.cu"))
        assert len(cu_files) >= 1, (
            f"Expected at least 1 .cu stub in {BLUEPRINT_DIR}, found {len(cu_files)}"
        )

        # --- No error in state ---
        assert final_state.get("error") is None, (
            f"Pipeline error: {final_state.get('error')}"
        )
