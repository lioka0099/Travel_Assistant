from __future__ import annotations
import os
from langgraph.graph import StateGraph, START, END
from .state import GraphState

# LangSmith imports
from langsmith import Client
from langchain_core.tracers import LangChainTracer

# Nodes
from .nodes import (
    route_intent,
    smalltalk,
    handler,
    resolve_place_llm,
    plan_tools,
    plan_time,
    clarify_missing,
    fetch_data,
    compose_answer,
    critique,
    revise,
    update_summary,
)

# ---------------------- Gate functions (conditions) ----------------------

def _after_route(state: GraphState) -> str:
    """If router chose smalltalk → smalltalk, else proceed to normal flow."""
    return "smalltalk" if state.get("intent") == "smalltalk" else "normal"

def _after_resolve(state: GraphState) -> str:
    """
    If resolve_place_llm set a 'final' message (ambiguous → disambiguation ask),
    end the turn by going to update_summary; otherwise continue to planning.
    """
    return "ask" if state.get("final") else "plan"

def _clarify_gate(state: GraphState) -> str:
    """
    If tools need a place (e.g., weather) but it's missing, ask a clarifying question.
    """
    data = state.get("data") or {}
    plan = data.get("plan") or {}
    place = plan.get("place") or data.get("resolved_place")
    if plan.get("weather") and not place:
        return "clarify"
    return "ok"

def _critique_gate(state: GraphState) -> str:
    """Send draft to critique only when critique_needed is true."""
    return "critique" if state.get("critique_needed") else "skip"

# ---------------------------- LangSmith helpers -----------------------------

def _get_langsmith_client() -> Client | None:
    """Initialize LangSmith client if API key is available."""
    api_key = os.getenv("LANGCHAIN_API_KEY")
    if api_key:
        return Client(api_key=api_key)
    return None

def _get_langsmith_tracer() -> LangChainTracer | None:
    """Get LangSmith tracer for tracing."""
    client = _get_langsmith_client()
    if client:
        return LangChainTracer(
            client=client,
            project_name=os.getenv("LANGCHAIN_PROJECT", "travel-assistant")
        )
    return None

# ---------------------------- Build function -----------------------------

def build_graph():
    g = StateGraph(GraphState)

    # Register nodes
    g.add_node("route", route_intent)
    g.add_node("smalltalk", smalltalk)
    g.add_node("handler", handler)
    g.add_node("resolve_place", resolve_place_llm)
    g.add_node("plan", plan_tools)
    g.add_node("plan_time", plan_time)
    g.add_node("clarify", clarify_missing)
    g.add_node("fetch", fetch_data)
    g.add_node("compose", compose_answer)
    g.add_node("critique", critique)
    g.add_node("revise", revise)
    g.add_node("update_summary", update_summary)

    # Start → Router
    g.add_edge(START, "route")

    # Router branch
    g.add_conditional_edges(
        "route",
        _after_route,
        {
            "smalltalk": "smalltalk",
            "normal": "handler",
        },
    )

    # Smalltalk path ends the turn
    g.add_edge("smalltalk", "update_summary")
    g.add_edge("update_summary", END)

    # Travel pipeline
    g.add_edge("handler", "resolve_place")

    # NEW: Early exit if resolver asks a disambiguation question
    g.add_conditional_edges(
        "resolve_place",
        _after_resolve,
        {
            "ask": "update_summary",   # 'final' set → show message & end turn
            "plan": "plan",
        },
    )

    g.add_edge("plan", "plan_time")

    # Clarify gate (blocking slot like missing place for weather)
    g.add_conditional_edges(
        "plan_time",
        _clarify_gate,
        {
            "clarify": "clarify",
            "ok": "fetch",
        },
    )

    # Clarify path ends the turn
    g.add_edge("clarify", "update_summary")

    # Fetch → Compose → (Critique?) → Revise → Summary
    g.add_edge("fetch", "compose")

    g.add_conditional_edges(
        "compose",
        _critique_gate,
        {
            "critique": "critique",
            "skip": "update_summary",
        },
    )
    g.add_edge("critique", "revise")
    g.add_edge("revise", "update_summary")

    # Compile the graph with optional LangSmith tracing
    compiled_graph = g.compile()
    
    # Add LangSmith tracer if available
    tracer = _get_langsmith_tracer()
    if tracer:
        compiled_graph = compiled_graph.with_config({"callbacks": [tracer]})
    
    return compiled_graph
