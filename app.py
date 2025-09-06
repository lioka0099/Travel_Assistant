import os
import streamlit as st
from graph import build_graph
from graph.state import GraphState
from graph.tools.location import get_client_location_data
from streamlit_js_eval import get_geolocation


# ---------- 1) App setup ----------
st.set_page_config(
    page_title="Travel Assistant", 
    page_icon="🧭", 
    layout="centered",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
    /* Theme-aware CSS variables */
    :root {
        --primary-gradient: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        --text-primary: var(--text-color);
        --bg-primary: var(--background-color);
        --bg-secondary: var(--secondary-background-color);
        --border-color: var(--border-color);
    }
    
    .main-header {
        text-align: center;
        padding: 1.5rem 0;
        background: var(--primary-gradient);
        color: white;
        border-radius: 10px;
        margin-bottom: 1rem;
    }
    
    .location-badge {
        background: var(--primary-gradient);
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        display: inline-block;
        margin: 0.5rem 0;
        text-align: center;
        width: 100%;
    }
    
    .welcome-section {
        text-align: center;
        margin: 1rem 0;
        padding: 1rem;
    }
    
    .welcome-section h1, .welcome-section h2 {
        color: var(--text-primary);
    }
    
    .features-row {
        display: flex;
        justify-content: space-around;
        flex-wrap: wrap;
        margin: 1rem 0;
        gap: 0.8rem;
    }
    
    .feature-item {
        display: flex;
        align-items: center;
        gap: 0.5rem;
        padding: 0.8rem 1rem;
        background: rgba(102, 126, 234, 0.1);
        color: var(--text-primary);
        border: 2px solid rgba(102, 126, 234, 0.3);
        border-radius: 12px;
        flex: 1;
        min-width: 200px;
        justify-content: center;
        transition: all 0.3s ease;
        font-weight: 500;
        backdrop-filter: blur(10px);
    }
    
    .feature-item:hover {
        transform: translateY(-3px);
        background: rgba(102, 126, 234, 0.2);
        border-color: rgba(102, 126, 234, 0.5);
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.2);
    }
    
    /* Dark mode specific feature item styling */
    .stApp[data-theme="dark"] .feature-item {
        background: rgba(102, 126, 234, 0.15);
        border-color: rgba(102, 126, 234, 0.4);
    }
    
    .stApp[data-theme="dark"] .feature-item:hover {
        background: rgba(102, 126, 234, 0.25);
        border-color: rgba(102, 126, 234, 0.6);
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.3);
    }
    
    .examples-section {
        background: var(--bg-secondary);
        color: var(--text-primary);
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        border: 1px solid var(--border-color);
    }
    
    .examples-section h3 {
        color: var(--text-primary);
        margin-top: 0;
    }
    
    .examples-section ul {
        color: var(--text-primary);
    }
    
    .examples-section li {
        margin: 0.3rem 0;
        padding: 0.3rem 0;
    }
    
    .chat-container {
        max-width: 800px;
        margin: 0 auto;
    }
    
    .stChatMessage {
        margin: 1rem 0;
    }
    
    /* Ensure text is readable in both themes */
    .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {
        color: var(--text-primary) !important;
    }
    
    /* Style the welcome card */
    .welcome-card {
        background: var(--bg-secondary);
        color: var(--text-primary);
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        border: 1px solid var(--border-color);
    }
    
    .welcome-card h3 {
        color: var(--text-primary);
        margin-top: 0;
    }
    
    .welcome-card ul {
        color: var(--text-primary);
        margin: 0;
        padding-left: 1.5rem;
    }
    
    .welcome-card li {
        margin: 0.5rem 0;
        line-height: 1.4;
    }
</style>
""", unsafe_allow_html=True)

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
    ("chat_started", False),      # Track if user has started chatting
]:
    if key not in st.session_state:
        st.session_state[key] = default

location = get_geolocation()

# Auto-detect user location on first load.
# We only call get_client_location_data when both coordinates are available.
if location and isinstance(location, dict) and "coords" in location:
    coords = location.get("coords") or {}
    lat = coords.get("latitude")
    lon = coords.get("longitude")
    with st.spinner("Detecting your location..."):
        if lat is not None and lon is not None:
            location_data = get_client_location_data(lat, lon)
        else:
            location_data = None
        if location_data:
            st.session_state.user_profile["current_location"] = location_data["location_string"]
            st.session_state.user_profile["location_data"] = location_data
            st.session_state["location_detected"] = True
        else:
            st.session_state["location_detected"] = True

# Show welcome page if chat hasn't started
if not st.session_state.chat_started:
    
    # Welcome message
    st.markdown("""
    <div class="welcome-section">
        <h1>🌍 Your personal travel assistant</h1>
        <h2>I can help you with:</h2>
    </div>
    """, unsafe_allow_html=True)
    
    # Features in a compact row
    st.markdown("""
    <div class="features-row">
        <div class="feature-item">🌤️ Weather Forecasts</div>
        <div class="feature-item">🎒 Packing Advice</div>
        <div class="feature-item">🏛️ Attractions & Activities</div>
        <div class="feature-item">🚗 Distance & Travel</div>
        <div class="feature-item">🌍 Country Information</div>
        <div class="feature-item">🔍 Web Search</div>
    </div>
    """, unsafe_allow_html=True)
    
    # Example prompts
    st.markdown("""
    <div class="welcome-card">
        <h3>Try asking me:</h3>
        <ul>
            <li>"What's the weather like in Paris today?"</li>
            <li>"Suggest a destination 2 hours away from me"</li>
            <li>"What should I pack for a trip to Tokyo in December?"</li>
            <li>"What are the top attractions in Rome?"</li>
            <li>"Compare the weather in Barcelona vs Madrid"</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

# Chat interface
st.markdown('<div class="chat-container">', unsafe_allow_html=True)

# ---------- 3) Chat transcript ----------
for msg in st.session_state.history:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# ---------- 4) Message input ----------
user_msg = st.chat_input("Ask about destinations, packing, attractions…")
if user_msg:
    # Mark chat as started
    st.session_state.chat_started = True
    
    # Show user's bubble immediately
    st.session_state.history.append({"role": "user", "content": user_msg})
    with st.chat_message("user"):
        st.write(user_msg)

    # Create a placeholder for the assistant message
    assistant_placeholder = st.empty()
    
    # Show thinking indicator
    with assistant_placeholder.container():
        with st.chat_message("assistant"):
            st.write("Thinking...")
    
    # Prepare graph input state
    state: GraphState = {
        "history": st.session_state.history,
        "user_msg": user_msg,
        "user_profile": st.session_state.user_profile,
        "summary": st.session_state.summary,
        "data": {**(st.session_state.data or {}), "web_allowed": True, "units": "metric"},
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

    # Get the final response
    assistant_text = out.get("final")
    if not assistant_text:
        assistant_text = out.get("draft", "(no reply)")

    # Replace the entire assistant message with the final response
    with assistant_placeholder.container():
        with st.chat_message("assistant"):
            st.write(assistant_text)

    # Add to history
    st.session_state.history.append({"role": "assistant", "content": assistant_text})

st.markdown('</div>', unsafe_allow_html=True)
