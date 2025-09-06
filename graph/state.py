from typing import TypedDict, List, Dict, Any, Optional

from langchain_core.runnables import history

class GraphState(TypedDict, total=False):
    """Shared state passed between graph nodes.

    total=False allows nodes to perform partial updates without specifying all fields.

    Keys:
    - history: list of chat messages {role, content}
    - user_msg: latest user message (normalized)
    - intent: routed intent label (e.g., weather, attractions)
    - user_profile: durable slots such as destination and dates
    - data: planning flags and fetched facts
    - draft: intermediate draft used for critique
    - final: finalized assistant reply
    - critique_needed: gate for critique step
    - critique_notes: critique output, if any
    - offtopic_counter: smalltalk redirection level
    - summary: compact, durable conversation summary
    """

    history: List[Dict[str, str]]
    user_msg: str
    intent: str
    user_profile: Dict[str, Any]
    data: Dict[str, Any]
    draft: str
    final: str
    critique_needed: bool
    critique_notes: Optional[str]
    offtopic_counter: int
    summary: str