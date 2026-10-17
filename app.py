import json
import re
import uuid
from urllib.parse import urlparse

import groq
import streamlit as st
from langchain_core.messages import HumanMessage
from langgraph.errors import GraphRecursionError
from streamlit_js_eval import get_geolocation

from graph import build_graph
from graph.tools.location import get_client_location_data
from llm.llm_client import make_agent_model, make_review_model

st.set_page_config(page_title="Travel Assistant", page_icon="🧭", layout="centered", initial_sidebar_state="collapsed")

EXAMPLES = [
    "Find me a weekend trip within 2 hours' drive with good weather",
    "What's the weather in Paris this weekend?",
    "Compare the weather in Barcelona and Madrid",
    "What should I pack for Tokyo next week?",
    "What currency and language do they use in Iceland?",
]

HELP = """**I can help you plan trips:**
- Weather forecasts, up to 16 days ahead
- Drive times and distances (by car only)
- Country facts: currency, language, capital, dialing code
- Attractions, events and ideas from the web
- Packing advice

For "near me" questions I use your browser location, if you allow it.
I only help with travel topics."""

# Status line shown while each tool runs (#1 visibility, #2 the user's language).
PROGRESS = {
    "get_weather": "Checking the weather in {place}…",
    "get_country_info": "Looking up {country}…",
    "web_search": "Searching the web…",
    "get_local_time": "Checking the time in {timezone}…",
    "get_user_location": "Finding your location…",
    "get_travel_time": "Checking drive time to {destination}…",
}
SOURCES = {"get_weather": "Open-Meteo", "get_country_info": "mledoze/countries", "get_travel_time": "OSRM"}

CSS = """
<style>
.welcome-section { text-align: center; margin: 1rem 0; padding: 1rem; }
.features-row { display: flex; justify-content: space-around; flex-wrap: wrap; margin: 1rem 0; gap: 0.8rem; }
.feature-item {
    display: flex; align-items: center; justify-content: center; gap: 0.5rem; flex: 1; min-width: 200px;
    padding: 0.8rem 1rem; border-radius: 12px; font-weight: 500;
    background: rgba(102, 126, 234, 0.1); border: 2px solid rgba(102, 126, 234, 0.3);
}
</style>
"""

WELCOME = """
<div class="welcome-section">
    <h1>🌍 Your personal travel assistant</h1>
    <h2>I can help you with:</h2>
</div>
<div class="features-row">
    <div class="feature-item">🌤️ Weather Forecasts</div>
    <div class="feature-item">🚗 Drive Times</div>
    <div class="feature-item">🏛️ Attractions & Activities</div>
    <div class="feature-item">🎒 Packing Advice</div>
    <div class="feature-item">🌍 Country Information</div>
    <div class="feature-item">🔍 Web Search</div>
</div>
"""


def md(text: str) -> str:
    """Make text safe for st.markdown: drop images, keeping their alt text (an image URL is fetched on
    render, so one injected via a web result could leak the user's data; the replace catches
    reference-style images), and escape $ so prices like '$20 to $30' aren't rendered as LaTeX."""
    text = re.sub(r"!\[([^\]]*)\]\([^)]*\)", r"\1", text).replace("![", "[")
    return text.replace("$", r"\$")


def plural(n: int, word: str) -> str:
    return f"{n} {word}{'' if n == 1 else 's'}"


def progress_line(tool_call: dict) -> str:
    try:
        return PROGRESS[tool_call["name"]].format(**tool_call["args"])
    except KeyError:
        return "Working…"


def friendly_error(e: Exception) -> str:
    """Plain-language error message (#9)."""
    if isinstance(e, groq.RateLimitError):
        return "The AI service is busy right now. Wait a moment and try again."
    if isinstance(e, groq.AuthenticationError):
        return "The AI service isn't set up correctly (API key problem)."
    if "model_decommissioned" in str(e) or "model_not_found" in str(e):
        return "The AI model this app uses is no longer available. The app needs its model setting updated."
    if isinstance(e, GraphRecursionError):
        return "I got stuck working on that one. Try rephrasing your question."
    return "Something went wrong on my side. Try again or start a new chat."


def apply_update(node: str, out: dict, turn: dict, status) -> None:
    """Fold one graph update into the turn being built and refresh the status line."""
    if node == "agent":
        msg = out["messages"][-1]
        status.markdown("⏳ " + md(progress_line(msg.tool_calls[0]) if msg.tool_calls else "Double-checking the answer…"))
    elif node == "tools":
        for m in out["messages"]:
            try:
                data = json.loads(m.content)
            except ValueError:
                data = {}
            if m.status == "error" or "summary" not in data:
                turn["steps"].append(f"Couldn't get data from {SOURCES.get(m.name, 'a service')}")
                continue
            turn["steps"].append(data["summary"])
            if m.name in SOURCES and SOURCES[m.name] not in turn["sources"]:
                turn["sources"].append(SOURCES[m.name])
            for r in data.get("results", []):
                link = f"[{urlparse(r['url']).netloc}]({r['url']})"
                if link not in turn["sources"]:
                    turn["sources"].append(link)
    elif node == "reviewer":
        if "final_answer" in out:
            turn["content"] = out["final_answer"]
            turn["checked"] = not out["review_issues"]
        else:
            turn["fixes"] += 1
            status.markdown("⏳ Fixing a detail…")


def render_assistant(turn: dict, last: bool) -> None:
    if "error" in turn:
        st.error(turn["error"], icon="⚠️")
        if last and st.button("Try again", key="retry"):
            st.session_state.chat.pop()  # drop the error; the user's message before it is re-sent
            st.session_state.pending = st.session_state.chat[-1]["content"]
            st.rerun()
        with st.expander("Technical details"):
            st.code(turn["details"], language=None)
        return
    st.markdown(md(turn["content"]))
    if not turn["steps"] and not turn["fixes"]:
        return  # greetings, declines, general advice: nothing to show (#8)
    check = "double-checked ✓" if turn["checked"] else "couldn't fully double-check ⚠️"
    with st.expander(f"How I got this · {plural(len(turn['steps']), 'step')} · {check}"):
        st.markdown(md("\n".join(f"{i}. {s}" for i, s in enumerate(turn["steps"], 1))))
        if turn["checked"]:
            fixed = f" (fixed {plural(turn['fixes'], 'detail')})" if turn["fixes"] else ""
            st.markdown(f"✓ Double-checked the answer against this data{fixed}")
        else:
            st.markdown("⚠️ Some details couldn't be double-checked; see the note in the answer.")
        if turn["sources"]:
            st.caption(md("Sources: " + " · ".join(turn["sources"])))


def submit(text: str) -> None:
    st.session_state.chat.append({"role": "user", "content": text})
    st.session_state.pending = text
    st.rerun()


def run_turn(text: str) -> None:
    turn = {"role": "assistant", "content": "", "steps": [], "sources": [], "checked": True, "fixes": 0}
    config = {
        "configurable": {"thread_id": st.session_state.thread_id, "user_location": st.session_state.get("location")},
        "recursion_limit": 40,
    }
    inputs = {"messages": [HumanMessage(text)], "review_count": 0, "review_issues": [], "final_answer": ""}
    with st.chat_message("assistant"):
        status = st.empty()
        status.markdown("⏳ Thinking…")
        try:
            for update in st.session_state.graph.stream(inputs, config, stream_mode="updates"):
                for node, out in update.items():
                    apply_update(node, out, turn, status)
        except Exception as e:  # any failure becomes a plain message with a way out (#9)
            turn = {"role": "assistant", "error": friendly_error(e), "details": f"{type(e).__name__}: {e}"}
        # Cleared only now: a Streamlit rerun (a BaseException) mid-turn leaves it set, so the turn runs again.
        st.session_state.pending = None
        status.empty()
        st.session_state.chat.append(turn)
        render_assistant(turn, last=True)


# ---------- Session setup ----------
st.markdown(CSS, unsafe_allow_html=True)

if "graph" not in st.session_state:
    try:
        st.session_state.graph = build_graph(make_agent_model(), make_review_model())
    except groq.GroqError as e:
        st.error("The AI service isn't set up: the GROQ_API_KEY setting is missing.", icon="⚠️")
        with st.expander("Technical details"):
            st.code(f"{type(e).__name__}: {e}", language=None)
        st.stop()
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())
st.session_state.setdefault("chat", [])
st.session_state.setdefault("pending", None)

# Browser location: requested and reverse-geocoded once per session.
if "location" not in st.session_state:
    geo = get_geolocation()  # None until the browser answers
    if geo:
        coords = geo.get("coords")
        st.session_state.location = get_client_location_data(coords["latitude"], coords["longitude"]) if coords else None

# ---------- Top row: location, help, new chat (#1, #3, #10) ----------
loc = st.session_state.get("location")
badge, help_col, new_col = st.columns([3, 2, 1.2], vertical_alignment="center")
if loc:
    badge.caption(f"📍 Using your location: {loc['city'] or loc['location_string']}")
with help_col.popover("❓ What can I ask?"):
    st.markdown(HELP)
if new_col.button("New chat"):
    st.session_state.chat = []
    st.session_state.thread_id = str(uuid.uuid4())
    st.session_state.pending = None
    st.rerun()

# ---------- Welcome screen (#5, #6, #10) ----------
if not st.session_state.chat:
    st.markdown(WELCOME, unsafe_allow_html=True)
    st.caption("Weather forecasts cover up to 16 days ahead. Drive times are by car only.")
    st.markdown("**Try asking:**")
    for example in EXAMPLES:
        if st.button(example, width="stretch"):
            submit(example)

# ---------- Chat ----------
for i, turn in enumerate(st.session_state.chat):
    with st.chat_message(turn["role"]):
        if turn["role"] == "user":
            st.markdown(md(turn["content"]))
        else:
            render_assistant(turn, last=i == len(st.session_state.chat) - 1)

if prompt := st.chat_input("Ask about destinations, weather, packing, drive times…", submit_mode="disable"):
    submit(prompt)

if st.session_state.pending:
    run_turn(st.session_state.pending)
