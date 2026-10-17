import logging
from datetime import date

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage, trim_messages
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode
from pydantic import BaseModel, Field

from .prompts import AGENT_SYSTEM, REVIEW_SYSTEM
from .tools import TOOLS

MAX_REVIEWS = 3  # first draft + up to 2 revisions
HISTORY_LIMIT = 30  # messages sent to the agent model; the checkpoint keeps everything
REVIEWER = "reviewer"  # name on feedback messages, so they aren't mistaken for the user's question

logger = logging.getLogger(__name__)


class Review(BaseModel):
    approved: bool = Field(description="True only if the draft passes every check")
    issues: list[str] = Field(description="Short plain-language problems to fix; empty when approved")


class AgentState(MessagesState):
    review_count: int
    review_issues: list[str]
    final_answer: str


def current_turn(messages: list[BaseMessage]) -> tuple[HumanMessage | None, list[BaseMessage]]:
    """The user's latest question and every message after it."""
    for i in range(len(messages) - 1, -1, -1):
        m = messages[i]
        if isinstance(m, HumanMessage) and m.name != REVIEWER:
            return m, messages[i + 1 :]
    return None, messages


def build_graph(agent_model: BaseChatModel, review_model: BaseChatModel):
    """Agent ⇄ tools loop, then a reviewer from another model family checks every draft."""
    agent_llm = agent_model.bind_tools(TOOLS)
    review_llm = review_model.with_structured_output(Review, method="json_schema", strict=True)

    def agent(state: AgentState) -> dict:
        system = SystemMessage(AGENT_SYSTEM.format(today=f"{date.today():%A, %Y-%m-%d}"))
        messages = state["messages"]
        _, turn = current_turn(messages)
        start = max(len(messages) - len(turn) - 1, 0)  # index of the user's latest question
        # A turn cut off mid-way (e.g. by a Streamlit rerun) can leave a tool call with no result,
        # which Groq rejects on every later call, so drop those.
        answered = {m.tool_call_id for m in messages[:start] if isinstance(m, ToolMessage)}
        earlier = [m for m in messages[:start] if not isinstance(m, AIMessage) or all(c["id"] in answered for c in m.tool_calls)]
        # Trim only earlier turns: the current question and its tool results must always reach the model.
        earlier = trim_messages(earlier, max_tokens=HISTORY_LIMIT, token_counter=len, start_on="human")
        return {"messages": [agent_llm.invoke([system, *earlier, *messages[start:]])]}

    def after_agent(state: AgentState) -> str:
        return "tools" if state["messages"][-1].tool_calls else "reviewer"

    def reviewer(state: AgentState) -> dict:
        question, turn = current_turn(state["messages"])
        draft = state["messages"][-1].content
        tool_results = "\n".join(m.content for m in turn if isinstance(m, ToolMessage)) or "(no tools were called)"
        previous = "\n".join(f"- {i}" for i in state.get("review_issues") or []) or "(first review)"
        count = state.get("review_count", 0) + 1
        try:
            review = review_llm.invoke(
                [
                    SystemMessage(REVIEW_SYSTEM),
                    HumanMessage(
                        f"Question:\n{question.content if question else ''}\n\n"
                        f"Tool results:\n{tool_results}\n\n"
                        f"Previous issues:\n{previous}\n\n"
                        f"Draft:\n{draft}"
                    ),
                ]
            )
        except Exception:  # reviewer unavailable (e.g. rate limit): show the draft, clearly marked as unchecked
            logger.warning("Review failed; showing the draft unchecked", exc_info=True)
            note = "\n\n⚠️ I couldn't double-check this answer."
            return {"review_count": count, "review_issues": ["review unavailable"], "final_answer": draft + note}
        if review.approved:
            return {"review_count": count, "review_issues": [], "final_answer": draft}
        issues = review.issues or ["parts of this answer"]
        if count >= MAX_REVIEWS:
            note = "\n\n⚠️ I couldn't double-check: " + "; ".join(issues)
            return {"review_count": count, "review_issues": issues, "final_answer": draft + note}
        feedback = HumanMessage(
            "Reviewer feedback. Fix these issues, then write your full corrected answer:\n"
            + "\n".join(f"- {i}" for i in issues),
            name=REVIEWER,
        )
        return {"review_count": count, "review_issues": issues, "messages": [feedback]}

    def after_review(state: AgentState) -> str:
        last = state["messages"][-1]
        return "agent" if isinstance(last, HumanMessage) and last.name == REVIEWER else END

    g = StateGraph(AgentState)
    g.add_node("agent", agent)
    g.add_node("tools", ToolNode(TOOLS, handle_tool_errors=True))  # LangGraph 1.x re-raises tool errors by default
    g.add_node("reviewer", reviewer)
    g.add_edge(START, "agent")
    g.add_conditional_edges("agent", after_agent, ["tools", "reviewer"])
    g.add_edge("tools", "agent")
    g.add_conditional_edges("reviewer", after_review, ["agent", END])
    return g.compile(checkpointer=InMemorySaver())
