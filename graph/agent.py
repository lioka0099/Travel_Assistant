import logging
from datetime import date

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage, trim_messages
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode
from pydantic import BaseModel, Field

from .prompts import AGENT_SYSTEM, BUDGET_NOTE, REVIEW_SYSTEM
from .tools import TOOLS

MAX_REVIEWS = 3  # first draft + up to 2 revisions
TOOL_BUDGET = 12  # tool results per question; after that the agent must answer with what it has
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


def today() -> str:
    return f"{date.today():%A, %Y-%m-%d}"


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
    # No strict=True: Groq only enforces it for gpt-oss models. One quick retry covers a malformed
    # review; Groq's client already backs off on rate limits.
    review_llm = review_model.with_structured_output(Review, method="json_schema").with_retry(
        stop_after_attempt=2, wait_exponential_jitter=False
    )

    def agent(state: AgentState) -> dict:
        system = SystemMessage(AGENT_SYSTEM.format(today=today()))
        messages = state["messages"]
        _, turn = current_turn(messages)
        start = max(len(messages) - len(turn) - 1, 0)  # index of the user's latest question
        # A turn cut off mid-way (e.g. by a Streamlit rerun) can leave a tool call with no result,
        # which Groq rejects on every later call, so drop those.
        answered = {m.tool_call_id for m in messages[:start] if isinstance(m, ToolMessage)}
        earlier = [m for m in messages[:start] if not isinstance(m, AIMessage) or all(c["id"] in answered for c in m.tool_calls)]
        # Trim only earlier turns: the current question and its tool results must always reach the model.
        earlier = trim_messages(earlier, max_tokens=HISTORY_LIMIT, token_counter=len, start_on="human")
        prompt = [system, *earlier, *messages[start:]]
        if sum(isinstance(m, ToolMessage) for m in turn) >= TOOL_BUDGET:  # no tools bound: it has to answer now
            return {"messages": [agent_model.invoke([*prompt, SystemMessage(BUDGET_NOTE)])]}
        return {"messages": [agent_llm.invoke(prompt)]}

    def after_agent(state: AgentState) -> str:
        return "tools" if state["messages"][-1].tool_calls else "reviewer"

    def reviewer(state: AgentState) -> dict:
        messages = state["messages"]
        question, turn = current_turn(messages)
        # The previous question and answer, so follow-ups like "and Sunday?" can be judged.
        earlier_question, earlier_turn = current_turn(messages[: max(len(messages) - len(turn) - 1, 0)])
        earlier_answer = next((m.content for m in reversed(earlier_turn) if isinstance(m, AIMessage) and not m.tool_calls), "")
        earlier = (
            f"User: {earlier_question.content}\nAssistant: {earlier_answer or '(no answer)'}"
            if earlier_question
            else "(none)"
        )
        draft = messages[-1].content
        tool_results = "\n".join(m.content for m in turn if isinstance(m, ToolMessage)) or "(no tools were called)"
        previous = "\n".join(f"- {i}" for i in state.get("review_issues") or []) or "(first review)"
        count = state.get("review_count", 0) + 1
        if not draft.strip():  # nothing to check, so don't spend a review call
            review = Review(approved=False, issues=["the answer was empty"])
        else:
            try:
                review = review_llm.invoke(
                    [
                        SystemMessage(REVIEW_SYSTEM.format(today=today())),
                        HumanMessage(
                            f"<earlier_chat>\n{earlier}\n</earlier_chat>\n\n"
                            f"<question>\n{question.content if question else ''}\n</question>\n\n"
                            f"<tool_results>\n{tool_results}\n</tool_results>\n\n"
                            f"<previous_issues>\n{previous}\n</previous_issues>\n\n"
                            f"<draft>\n{draft}\n</draft>"
                        ),
                    ]
                )
            except Exception as e:  # reviewer unavailable (e.g. rate limit): show the draft, clearly marked as unchecked
                logger.error("Review failed (%s); showing the draft unchecked", type(e).__name__, exc_info=True)
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
