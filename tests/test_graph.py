import json
from datetime import date

import pytest
import requests
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_groq import ChatGroq

from fakes import ScriptedModel
from graph import build_graph
from graph.agent import MAX_REVIEWS, Review
from graph.tools import TOOLS
from graph.tools.http import session
from llm.llm_client import AGENT_MODEL, REVIEW_MODEL, make_agent_model, make_review_model

TEL_AVIV = {
    "location_string": "Tel Aviv, Israel", "city": "Tel Aviv", "region": "Tel Aviv District",
    "country": "Israel", "latitude": 32.08, "longitude": 34.78,
}


def call(name, args, id="call_1"):
    return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": id}])


def ok():
    return Review(approved=True, issues=[])


def reject(*issues):
    return Review(approved=False, issues=list(issues))


def config(thread="t1", location=None):
    return {"configurable": {"thread_id": thread, "user_location": location}, "recursion_limit": 40}


def inputs(text):
    return {"messages": [HumanMessage(text)], "review_count": 0, "review_issues": [], "final_answer": ""}


def run(graph, text, thread="t1", location=None):
    return graph.invoke(inputs(text), config(thread, location))


def test_answer_without_tools_is_approved_first_time():
    agent = ScriptedModel(script=[AIMessage("Pack layers and a rain jacket.")])
    reviewer = ScriptedModel(script=[ok()])
    out = run(build_graph(agent, reviewer), "What should I pack for Scotland?")
    assert out["final_answer"] == "Pack layers and a rain jacket."
    assert out["review_count"] == 1


def test_tool_result_reaches_agent_and_reviewer():
    agent = ScriptedModel(script=[call("get_user_location", {}), AIMessage("You're in Tel Aviv.")])
    reviewer = ScriptedModel(script=[ok()])
    out = run(build_graph(agent, reviewer), "Where am I?", location=TEL_AVIV)
    tool_msg = next(m for m in out["messages"] if isinstance(m, ToolMessage))
    assert json.loads(tool_msg.content)["summary"] == "Found your location: Tel Aviv, Israel"
    assert "Found your location: Tel Aviv, Israel" in reviewer.calls[0][1].content
    assert out["final_answer"] == "You're in Tel Aviv."


def test_rejected_draft_is_revised_and_rereviewed():
    agent = ScriptedModel(script=[AIMessage("Draft 1"), AIMessage("Draft 2")])
    reviewer = ScriptedModel(script=[reject("drive time for Akko not checked"), ok()])
    out = run(build_graph(agent, reviewer), "Weekend trip ideas?")
    assert out["final_answer"] == "Draft 2"
    assert out["review_count"] == 2
    feedback = agent.calls[1][-1]
    assert feedback.name == "reviewer" and "drive time for Akko not checked" in feedback.content
    assert "drive time for Akko not checked" in reviewer.calls[1][1].content  # second review checks the fix


def test_review_loop_stops_at_limit_with_a_note():
    agent = ScriptedModel(script=[AIMessage(f"Draft {i}") for i in range(1, MAX_REVIEWS + 1)])
    reviewer = ScriptedModel(script=[reject("opening hours for the Louvre") for _ in range(MAX_REVIEWS)])
    out = run(build_graph(agent, reviewer), "When does the Louvre open?")
    assert out["review_count"] == MAX_REVIEWS
    assert out["final_answer"] == f"Draft {MAX_REVIEWS}\n\n⚠️ I couldn't double-check: opening hours for the Louvre"


def test_tool_error_goes_back_to_agent_instead_of_crashing(monkeypatch):
    def down(*args, **kwargs):
        raise requests.ConnectionError("weather service down")

    monkeypatch.setattr(session, "get", down)
    agent = ScriptedModel(script=[call("get_weather", {"place": "Paris"}), AIMessage("I couldn't check the weather.")])
    reviewer = ScriptedModel(script=[ok()])
    out = run(build_graph(agent, reviewer), "Weather in Paris?")
    tool_msg = next(m for m in out["messages"] if isinstance(m, ToolMessage))
    assert tool_msg.status == "error"
    assert out["final_answer"] == "I couldn't check the weather."


def test_next_turn_works_after_a_failed_turn():
    agent = ScriptedModel(script=[RuntimeError("Groq is down"), AIMessage("Hi! Where are you headed?")])
    reviewer = ScriptedModel(script=[ok()])
    graph = build_graph(agent, reviewer)
    with pytest.raises(RuntimeError):
        run(graph, "Hello")
    out = run(graph, "Hello")
    assert out["final_answer"] == "Hi! Where are you headed?"


def test_new_turn_resets_review_state():
    agent = ScriptedModel(script=[AIMessage("A1"), AIMessage("A2"), AIMessage("B1")])
    reviewer = ScriptedModel(script=[reject("x"), ok(), ok()])
    graph = build_graph(agent, reviewer)
    run(graph, "first question")
    out = run(graph, "second question")
    assert out["review_count"] == 1
    assert "(first review)" in reviewer.calls[2][1].content
    assert "second question" in reviewer.calls[2][1].content


def test_long_turn_keeps_the_question_in_agent_input():
    # 18 tool calls make the turn longer than HISTORY_LIMIT; the question must still reach the agent.
    calls = [call("get_local_time", {"timezone": "Asia/Tokyo"}, id=f"call_{i}") for i in range(18)]
    agent = ScriptedModel(script=[*calls, AIMessage("Done.")])
    reviewer = ScriptedModel(script=[ok()])
    run(build_graph(agent, reviewer), "Plan my trip")
    last_input = agent.calls[-1]
    assert any(isinstance(m, HumanMessage) and m.content == "Plan my trip" for m in last_input)
    assert sum(isinstance(m, ToolMessage) for m in last_input) == 18


def test_reviewer_failure_shows_draft_marked_unchecked():
    agent = ScriptedModel(script=[AIMessage("Pack layers.")])
    reviewer = ScriptedModel(script=[RuntimeError("rate limited"), RuntimeError("rate limited")])  # fails the retry too
    out = run(build_graph(agent, reviewer), "What should I pack?")
    assert out["final_answer"] == "Pack layers.\n\n⚠️ I couldn't double-check this answer."
    assert out["review_issues"] == ["review unavailable"]


def test_interrupted_turn_leaves_no_unanswered_tool_call():
    # A Streamlit rerun can cut a turn off right after the agent asks for a tool. Groq rejects a
    # history with an unanswered tool call, so the next turn must not send it.
    agent = ScriptedModel(script=[call("get_user_location", {}), AIMessage("You're in Tel Aviv.")])
    reviewer = ScriptedModel(script=[ok()])
    graph = build_graph(agent, reviewer)
    for _ in graph.stream(inputs("Where am I?"), config(location=TEL_AVIV), stream_mode="updates"):
        break  # stop right after the agent's tool-call update
    assert graph.get_state(config()).next == ("tools",)
    out = run(graph, "Where am I?", location=TEL_AVIV)
    answered = {m.tool_call_id for m in agent.calls[-1] if isinstance(m, ToolMessage)}
    asked = [c["id"] for m in agent.calls[-1] if isinstance(m, AIMessage) for c in m.tool_calls]
    assert all(i in answered for i in asked)
    assert out["final_answer"] == "You're in Tel Aviv."


def test_review_is_retried_once():
    # e.g. the reviewer returned malformed JSON the first time
    agent = ScriptedModel(script=[AIMessage("Pack layers.")])
    reviewer = ScriptedModel(script=[ValueError("malformed review"), ok()])
    out = run(build_graph(agent, reviewer), "What should I pack?")
    assert out["final_answer"] == "Pack layers."
    assert out["review_issues"] == []
    assert len(reviewer.calls) == 2


def test_real_models_get_tools_and_a_json_schema(monkeypatch):
    # The fakes ignore binding, so check what the real Groq models would be sent, without calling Groq.
    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    monkeypatch.delenv("GROQ_MODEL", raising=False)
    monkeypatch.delenv("GROQ_REVIEW_MODEL", raising=False)
    sent = []

    def fake_generate(self, messages, stop=None, run_manager=None, **kwargs):
        sent.append((self.model_name, kwargs))
        reply = '{"approved": true, "issues": []}' if "response_format" in kwargs else "Pack layers."
        return ChatResult(generations=[ChatGeneration(message=AIMessage(reply))])

    monkeypatch.setattr(ChatGroq, "_generate", fake_generate)
    out = run(build_graph(make_agent_model(), make_review_model()), "What should I pack?")
    (agent_model, agent_kwargs), (review_model, review_kwargs) = sent
    assert agent_model == AGENT_MODEL and review_model == REVIEW_MODEL
    assert [t["function"]["name"] for t in agent_kwargs["tools"]] == [t.name for t in TOOLS]
    response_format = review_kwargs["response_format"]
    assert response_format["type"] == "json_schema"
    assert "strict" not in response_format["json_schema"]  # Groq only honours strict for gpt-oss models
    assert out["final_answer"] == "Pack layers."


def test_reviewer_sees_the_date_and_the_previous_turn():
    # Follow-ups like "and Sunday?" need the previous turn; forecast dates need today's date.
    agent = ScriptedModel(script=[AIMessage("Saturday in Paris looks sunny."), AIMessage("Sunday looks rainy.")])
    reviewer = ScriptedModel(script=[ok(), ok()])
    graph = build_graph(agent, reviewer)
    run(graph, "Weather in Paris on Saturday?")
    run(graph, "and Sunday?")
    system, prompt = reviewer.calls[1]
    assert f"{date.today():%A, %Y-%m-%d}" in system.content
    assert (
        "<earlier_chat>\nUser: Weather in Paris on Saturday?\nAssistant: Saturday in Paris looks sunny.\n</earlier_chat>"
        in prompt.content
    )


def test_reviewer_input_sections_are_tagged():
    # Tags keep untrusted tool results and the draft apart from the question and the review rules.
    agent = ScriptedModel(script=[call("get_user_location", {}), AIMessage("You're in Tel Aviv.")])
    reviewer = ScriptedModel(script=[ok()])
    run(build_graph(agent, reviewer), "Where am I?", location=TEL_AVIV)
    prompt = reviewer.calls[0][1].content
    assert "<question>\nWhere am I?\n</question>" in prompt
    assert "<tool_results>\n" in prompt and "Found your location: Tel Aviv, Israel" in prompt and "\n</tool_results>" in prompt
    assert "<previous_issues>\n(first review)\n</previous_issues>" in prompt
    assert prompt.endswith("<draft>\nYou're in Tel Aviv.\n</draft>")


def test_empty_draft_is_sent_back_without_a_review_call():
    agent = ScriptedModel(script=[AIMessage(""), AIMessage("Pack layers.")])
    reviewer = ScriptedModel(script=[ok()])
    out = run(build_graph(agent, reviewer), "What should I pack?")
    assert out["final_answer"] == "Pack layers."
    assert len(reviewer.calls) == 1
    assert "the answer was empty" in agent.calls[1][-1].content
