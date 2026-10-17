from langchain_core.messages import AIMessage
from streamlit.runtime.scriptrunner_utils.exceptions import RerunException
from streamlit.runtime.scriptrunner_utils.script_requests import RerunData
from streamlit.testing.v1 import AppTest

from fakes import ScriptedModel
from graph import build_graph
from graph.agent import Review

TEL_AVIV = {
    "location_string": "Tel Aviv, Israel", "city": "Tel Aviv", "region": "Tel Aviv District",
    "country": "Israel", "latitude": 32.08, "longitude": 34.78,
}


def start(agent_script, review_script):
    at = AppTest.from_file("../app.py", default_timeout=30)
    at.session_state["graph"] = build_graph(ScriptedModel(script=agent_script), ScriptedModel(script=review_script))
    at.session_state["location"] = TEL_AVIV  # skip the browser geolocation component
    return at.run()


def test_welcome_screen_shows_clickable_examples():
    at = start([], [])
    assert not at.exception
    assert "Find me a weekend trip within 2 hours' drive with good weather" in [b.label for b in at.button]
    assert "Tel Aviv" in at.caption[0].value


def test_tool_steps_show_in_plain_language():
    location_call = AIMessage(content="", tool_calls=[{"name": "get_user_location", "args": {}, "id": "c1"}])
    at = start([location_call, AIMessage("You're in Tel Aviv.")], [Review(approved=True, issues=[])])
    at.chat_input[0].set_value("Where am I?").run()
    assert not at.exception
    assert at.chat_message[-1].markdown[0].value == "You're in Tel Aviv."
    expander = at.expander[-1]
    assert expander.label == "How I got this · 1 step · double-checked ✓"
    assert "Found your location: Tel Aviv, Israel" in expander.markdown[0].value


def test_greeting_has_no_steps_box():
    at = start([AIMessage("Hi! Where would you like to go?")], [Review(approved=True, issues=[])])
    at.chat_input[0].set_value("hi").run()
    assert not at.exception
    assert len(at.expander) == 0


def test_error_shows_friendly_message_and_retry_works():
    at = start([RuntimeError("boom"), AIMessage("Hello again!")], [Review(approved=True, issues=[])])
    at.chat_input[0].set_value("hello").run()
    assert at.error[0].value == "Something went wrong on my side. Try again or start a new chat."
    at.button(key="retry").click().run()
    assert not at.error
    assert at.chat_message[-1].markdown[0].value == "Hello again!"
    assert [t["role"] for t in at.session_state["chat"]] == ["user", "assistant"]


def test_new_chat_clears_conversation():
    at = start([AIMessage("Hi!")], [Review(approved=True, issues=[])])
    at.chat_input[0].set_value("hi").run()
    old_thread = at.session_state["thread_id"]
    next(b for b in at.button if b.label == "New chat").click().run()
    assert at.session_state["chat"] == []
    assert at.session_state["thread_id"] != old_thread


def test_turn_interrupted_by_a_rerun_is_finished_on_the_next_run():
    # e.g. the browser answering the location prompt reruns the script in the middle of a turn
    location_call = AIMessage(content="", tool_calls=[{"name": "get_user_location", "args": {}, "id": "c1"}])
    rerun = RerunException(RerunData())
    at = start([location_call, rerun, AIMessage("You're in Tel Aviv.")], [Review(approved=True, issues=[])])
    at.chat_input[0].set_value("Where am I?").run()
    assert not at.exception
    assert [t["role"] for t in at.session_state["chat"]] == ["user", "assistant"]
    assert at.chat_message[-1].markdown[0].value == "You're in Tel Aviv."
