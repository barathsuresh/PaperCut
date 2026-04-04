import json
from typing import Any, Dict

from backend.tools.gemini_client import get_pro_model

_SYSTEM_PREFIX = """\
You are an expert ML research assistant. You have been given the context of a processed research paper.
Use this context to answer the user's question accurately and helpfully.

Paper Context:
{context}

User Question:
"""


def handle_chat(session_data: Dict[str, Any], message: str) -> str:
    context_parts = []

    node1_result = session_data.get("node1_result")
    if node1_result:
        contract_text = (
            json.dumps(node1_result, indent=2)
            if isinstance(node1_result, dict)
            else node1_result
        )
        context_parts.append(f"Research Contract:\n{contract_text}")

    node2_result = session_data.get("node2_result")
    if node2_result and node2_result.get("status") != "not_implemented":
        context_parts.append(f"Scaffold Code:\n{node2_result}")

    node3_result = session_data.get("node3_result")
    if node3_result and node3_result.get("status") != "not_implemented":
        context_parts.append(f"CUDA Blueprint:\n{json.dumps(node3_result, indent=2)}")

    context = (
        "\n\n".join(context_parts)
        if context_parts
        else "No paper has been processed yet for this session."
    )

    prompt = _SYSTEM_PREFIX.format(context=context) + message
    model = get_pro_model()
    response = model.generate_content(prompt)
    return response.text
