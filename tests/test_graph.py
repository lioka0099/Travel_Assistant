import json

import pytest
import requests
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from fakes import ScriptedModel
from graph import build_graph
from graph.agent import MAX_REVIEWS, Review
from graph.tools.http import session

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
    reviewer = ScriptedModel(script=[RuntimeError("rate limited")])
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
