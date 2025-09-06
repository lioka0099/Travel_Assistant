import os
import streamlit as st
from graph import build_graph
from graph.state import GraphState
from graph.tools.location import get_location_from_ip

# ---------- 1) App setup ----------
st.set_page_config(page_title="Travel Assistant", page_icon="🧭", layout="centered")
st.title("🧭 Travel Assistant")

# Lazy-build the compiled graph and stash it in the session
if "graph" not in st.session_state:
    st.session_state.graph = build_graph()

# ---------- 2) Session state (persist across turns) ----------
for key, default in [
    ("history", []),              # list[dict]: {"role": "user"/"assistant", "content": "..."}
    ("intent", None),             # last intent (for sticky-travel rule)
    ("offtopic_count", 0),        # consecutive smalltalk turns
    ("summary", ""),              # running TL;DR
    ("user_profile", {}),         # destinations MRU, dates, style, etc.
    ("data", {}),                 # tool facts, caches, flags (web_allowed, units), etc.
    ("web_allowed", True),        # UI toggle
]:
    if key not in st.session_state:
        st.session_state[key] = default

# Auto-detect user location on first load
if "location_detected" not in st.session_state:
    with st.spinner("Detecting your location..."):
        location_data = get_location_from_ip()
        if location_data:
            st.session_state.user_profile["current_location"] = location_data["location_string"]
            st.session_state.user_profile["location_data"] = location_data
            st.session_state["location_detected"] = True
            st.success(f"🧭 Detected location: {location_data['location_string']}")
        else:
            st.session_state["location_detected"] = True
            st.warning("Could not detect your location. You can still use the travel assistant!")

# ---------- 3) Controls ----------
col1, col2, col3 = st.columns([1,1,1])
with col1:
    st.session_state.web_allowed = st.toggle("Allow web (Tavily)", value=st.session_state.web_allowed)
with col2:
    # Add location input
    current_location = st.text_input("Your location", value=st.session_state.user_profile.get("current_location", ""), placeholder="e.g., New York, NY")
    if current_location and current_location != st.session_state.user_profile.get("current_location", ""):
        st.session_state.user_profile["current_location"] = current_location
with col3:
    if st.button("Reset conversation", type="secondary"):
        for k in ["history", "intent", "offtopic_count", "summary", "user_profile", "data"]:
            st.session_state[k] = [] if k=="history" else {} if k in ("user_profile","data") else 0 if k=="offtopic_count" else ""
        st.toast("Conversation cleared.", icon="🧹")

# Show detected location
if st.session_state.user_profile.get("current_location"):
    st.info(f"🧭 Your location: {st.session_state.user_profile['current_location']}")

# ---------- 4) Chat transcript ----------
for msg in st.session_state.history:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# ---------- 5) Message input ----------
user_msg = st.chat_input("Ask about destinations, packing, attractions…")
if user_msg:
    # Show user's bubble immediately
    st.session_state.history.append({"role": "user", "content": user_msg})
    with st.chat_message("user"):
        st.write(user_msg)

    # Prepare graph input state (include previous intent/summary/data to persist context)
    state: GraphState = {
        "history": st.session_state.history,
        "user_msg": user_msg,
        "user_profile": st.session_state.user_profile,
        "summary": st.session_state.summary,
        "data": {**(st.session_state.data or {}), "web_allowed": st.session_state.web_allowed, "units": "metric"},
        "intent": st.session_state.intent,
        "offtopic_count": st.session_state.offtopic_count,
    }

    # Run the graph
    out = st.session_state.graph.invoke(state)

    # Persist fields across turns
    st.session_state.intent = out.get("intent", st.session_state.intent)
    st.session_state.offtopic_count = out.get("offtopic_count", st.session_state.offtopic_count)
    st.session_state.summary = out.get("summary", st.session_state.summary)
    st.session_state.user_profile = out.get("user_profile", st.session_state.user_profile)
    st.session_state.data = out.get("data", st.session_state.data)

    # Decide assistant text: prefer 'final' (end-of-turn), else if compose produced a draft then later nodes set 'final'
    assistant_text = out.get("final")
    if not assistant_text:
        # some paths produce only 'draft' then critique/revise into 'final' inside the same run
        assistant_text = out.get("draft", "(no reply)")

    # Append and render assistant message
    st.session_state.history.append({"role": "assistant", "content": assistant_text})
    with st.chat_message("assistant"):
        st.write(assistant_text)

# ---------- 6) Debug panel (optional) ----------
with st.expander("🔎 Debug"):
    st.json({
        "intent": st.session_state.intent,
        "offtopic_count": st.session_state.offtopic_count,
        "summary": st.session_state.summary,
        "user_profile": st.session_state.user_profile,
        "data_keys": list(st.session_state.data.keys()) if isinstance(st.session_state.data, dict) else None,
    })
