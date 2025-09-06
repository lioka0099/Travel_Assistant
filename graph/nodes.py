from __future__ import annotations
from typing import Dict, Any, Optional, List, Tuple
from datetime import date, timedelta

from .state import GraphState
from llm.llm_client import (
    chat_completion_simple,
    chat_completion_structured,
    ToolPlan,
    ComposeOut,
    TimePlan,
    PlacePlan,
)
from .prompts import (
    STRICT_FACTS_POLICY, SYSTEM_PROMPT, ROUTER_PROMPT, COMPOSE_TMPL, SUMMARY_TMPL, REASONING_CHECKLIST,
    SMALLTALK_REDIRECT_PROMPT, PLANNER_SYS, TIME_PLANNER_SYS, PLACE_RESOLVER_SYS
)
from .policies import hint_weather, hint_country_facts, hint_web_search
from .tools.clock import now_iso, today
from .tools.weather import geocode, forecast_daily
from .tools.countries import country_facts
from .tools.tavily import web_search
from .tools.location import get_location_from_ip, geocode_location, calculate_distance, get_travel_time_estimate

#helpers
from .helpers.merge import deep_merge
from .helpers.destinations import remember_place, resolve_place, resolve_country_and_city
from .helpers.timeplan import resolve_relative_dates, weekend_for_country

# ------------------------------- nodes ----------------------------------

def route_intent(state: GraphState) -> Dict[str, Any]:
    msg = state["user_msg"].strip()
    intent = chat_completion_simple(
        [
            {"role": "system", "content": "Return exactly one word intent."},
            {"role": "user", "content": ROUTER_PROMPT.format(user_msg=msg)},
        ],
        temperature=0.0,
    ).strip().lower()

    recent_intent = state.get("intent")
    
    # Special handling for weather follow-ups
    if intent == "smalltalk" and recent_intent == "weather" and len(msg.split()) <= 3:
        intent = "weather"
    elif intent == "smalltalk" and recent_intent and recent_intent != "smalltalk" and len(msg.split()) <= 3:
        intent = recent_intent

    new_count = state.get("offtopic_count", 0)
    if intent != "smalltalk":
        new_count = 0
    return {"intent": intent, "offtopic_count": new_count}

def next_travel_question(profile: Dict[str, Any]) -> str:
    if not profile.get("destination"):
        return "Where are you thinking of traveling?"
    if not profile.get("start_date"):
        return f"When are you planning to go to {profile['destination']}?"
    if not profile.get("end_date"):
        return "How many days will you have?"
    if not profile.get("style"):
        return "Do you prefer nature, cities, or a mix?"
    return "What would you like help with first—destinations, packing, or things to do?"

def current_trip_context(state: GraphState) -> dict | None:
    profile = state.get("user_profile", {}) or {}
    dest = profile.get("active_destination") or profile.get("destination")
    if not dest:
        return None
    return {"destination": dest, "start": profile.get("start_date"), "end": profile.get("end_date")}

def smalltalk(state: GraphState) -> Dict[str, Any]:
    profile = state.get("user_profile", {}) or {}
    ctx = current_trip_context(state)
    if ctx:
        if ctx.get("start") and ctx.get("end"):
            question = f"Do you want me to keep planning for {ctx['destination']} ({ctx['start']} → {ctx['end']})?"
        elif ctx.get("start"):
            question = f"Should I keep planning for {ctx['destination']} starting {ctx['start']}?"
        else:
            question = f"Should I keep going with {ctx['destination']}?"
    else:
        question = next_travel_question(profile)

    reply = chat_completion_simple(
        [
            {"role": "system", "content": "You are friendly, brief, and helpful."},
            {"role": "user", "content": SMALLTALK_REDIRECT_PROMPT.format(user_msg=state["user_msg"], question=question)},
        ],
        temperature=0.5,
    )

    count = int(state.get("offtopic_count", 0)) + 1
    if count >= 3:
        reply += "\n\n(If you’re not planning a trip yet, that’s okay—tell me and I’ll share a few travel-idea prompts you can use anytime.)"
    return {"final": reply, "offtopic_count": count}

def handler(state: GraphState) -> Dict[str, Any]:
    msg = " ".join(state["user_msg"].split()).strip()
    short_hist = (state.get("history") or [])[-12:]
    data = dict(state.get("data") or {})
    data.setdefault("web_allowed", True)
    data.setdefault("units", "metric")
    return {"user_msg": msg, "history": short_hist, "data": data}

def resolve_place_llm(state: GraphState) -> dict:
    if state.get("intent") == "smalltalk":
        return {}
    profile = state.get("user_profile") or {}
    msg = state["user_msg"]
    active = profile.get("active_destination")
    dests = profile.get("destinations", [])

    plan: PlacePlan = chat_completion_structured(
        [
            {"role": "system", "content": PLACE_RESOLVER_SYS},
            {"role": "user", "content": (
                f"message: {msg}\n"
                f"active_destination: {active}\n"
                f"destinations: {dests}\n"
                "Return the structured fields."
            )},
        ],
        schema=PlacePlan,
        temperature=0.1,
    )

    data = dict(state.get("data") or {})
    if not plan.ambiguous and plan.resolved_place:
        data["resolved_place"] = plan.resolved_place
        upd = remember_place(state, plan.resolved_place)
        return deep_merge({"data": data}, upd)

    if plan.ambiguous and plan.alternatives:
        choices = "\n".join(f"{i+1}) {name}" for i, name in enumerate(plan.alternatives[:3]))
        q = "Did you mean:\n" + choices + "\n\nReply with the number or the exact name."
        return {"final": q, "data": deep_merge(data, {"place_candidates": plan.alternatives})}

    return {"data": data}

def resolve_user_location(state: GraphState) -> Dict[str, Any]:
    """Resolve and geocode the user's current location."""
    profile = state.get("user_profile", {})
    
    # Check if we already have location data
    if "location_data" in profile:
        return {}
    
    # Try to get location from IP if not already detected
    location_data = get_location_from_ip()
    if location_data:
        profile["current_location"] = location_data["location_string"]
        profile["location_data"] = location_data
        return {"user_profile": profile}
    
    return {}

def _is_weather_followup(state: Dict[str, Any], msg: str) -> bool:
    m = (msg or "").lower().strip()
    
    # Simple acknowledgments should NOT be treated as weather follow-ups
    simple_acks = ["thanks", "thank you", "ok", "okay", "got it", "perfect", "great", "awesome"]
    if m in simple_acks:
        return False
    
    # Check for time-related weather follow-ups
    time_words = any(w in m for w in ["weekend","today","tomorrow","tonight","morning","evening","afternoon","next week","this week"])
    facts = ((state.get("data") or {}).get("facts") or {})
    has_recent_weather = bool((facts.get("weather_by_place") or {}))
    last_assistant = next((h for h in reversed(state.get("history") or []) if h.get("role")=="assistant"), None)
    said_weather = "weather" in (last_assistant.get("content","").lower() if last_assistant else "")
    
    return time_words and (has_recent_weather or said_weather)

def plan_tools(state: GraphState) -> Dict[str, Any]:
    intent = state.get("intent", "")
    profile = state.get("user_profile", {}) or {}
    summary = state.get("summary", "")
    msg = state["user_msg"]
    data_block = state.get("data") or {}
    web_allowed = data_block.get("web_allowed", True)

    plan: ToolPlan = chat_completion_structured(
        [
            {"role": "system", "content": PLANNER_SYS},
            {"role": "user", "content": f"Intent: {intent}\nUser message: {msg}\nProfile: {profile}\nSummary: {summary}\nReturn booleans and a brief rationale."},
        ],
        schema=ToolPlan,
        temperature=0.1,
    )

    llm_resolved = data_block.get("resolved_place")
    place = llm_resolved or resolve_place(state) or plan.place_hint

    need_weather = plan.need_weather or hint_weather(msg)
    need_country = plan.need_country or hint_country_facts(msg)
    need_web = (plan.need_web or hint_web_search(msg)) and web_allowed

    if not need_weather and _is_weather_followup(state, msg):
        need_weather = True

    # Check if user is asking about distance-based recommendations
    msg_lower = msg.lower()
    distance_queries = ["hours away", "distance", "near me", "close to me", "nearby", "from here"]
    has_distance_query = any(phrase in msg_lower for phrase in distance_queries)
    
    # If asking about distance, we need location data
    need_location = has_distance_query and not (state.get("user_profile", {}).get("location_data"))

    print(f"DEBUG: msg='{msg}'")
    print(f"DEBUG: plan.need_weather={plan.need_weather}, hint_weather={hint_weather(msg)}, final need_weather={need_weather}")
    print(f"DEBUG: place={place}")
    print(f"DEBUG: llm_resolved={llm_resolved}")
    print(f"DEBUG: resolve_place(state)={resolve_place(state)}")
    print(f"DEBUG: has_distance_query={has_distance_query}, need_location={need_location}")

    data_plan = {"weather": need_weather, "country": need_country, "web": need_web, "place": place, "location": need_location}
    new_data = deep_merge(data_block, {"plan": data_plan, "web_allowed": web_allowed})
    return {"data": new_data}

def plan_time(state: GraphState) -> Dict[str, Any]:
    data = dict(state.get("data") or {})
    plan = data.get("plan") or {}
    if not (plan.get("weather") or plan.get("web")):
        data["time_plan"] = {"target_type": "unspecified"}
        return {"data": data}

    intent = state.get("intent", "")
    profile = state.get("user_profile", {}) or {}
    msg = state["user_msg"]

    tp: TimePlan = chat_completion_structured(
        [
            {"role": "system", "content": TIME_PLANNER_SYS},
            {"role": "user", "content": (
                f"Intent: {intent}\n"
                f"Message: {msg}\n"
                f"Profile: {profile}\n"
                "If the message mentions a time-of-day (tonight/evening/morning/afternoon) without a date, "
                "map it to TODAY (destination timezone). Return structured fields."
            )},
        ],
        schema=TimePlan,
        temperature=0.1,
    )

    data["time_plan"] = tp.model_dump()
    return {"data": data}

def _needs_hard_clarification(plan: dict, state: GraphState) -> Tuple[bool, Optional[str]]:
    data = state.get("data") or {}
    resolved = data.get("resolved_place")
    place = plan.get("place") or resolved
    if plan.get("weather") and not place:
        return True, "Which city or area should I check the weather for?"
    return False, None

def clarify_missing(state: GraphState) -> dict:
    plan = (state.get("data") or {}).get("plan") or {}
    need, q = _needs_hard_clarification(plan, state)
    if need and q:
        return {"final": "I can do that. " + q}
    return {}

def fetch_data(state: GraphState) -> Dict[str, Any]:
    data_in = state.get("data") or {}
    plan = data_in.get("plan") or {}

    facts = {"now": now_iso(), "today": today(), "weather_by_place": {}}
    profile_update: Dict[str, Any] = {}

    place_to_use = plan.get("place") or data_in.get("resolved_place")
    place_tz = None

    country_code = None

    if plan.get("weather") and place_to_use:
        print(f"DEBUG: Attempting to fetch weather for: {place_to_use}")
        print(f"DEBUG: plan.get('weather') = {plan.get('weather')}")
        g = geocode(place_to_use)
        print(f"DEBUG: geocode result: {g}")
        country_code = g.get("country_code") or g.get("country") if g else None
        print(f"DEBUG: country_code from geocode: {country_code}")
        if g:
            units = data_in.get("units", "metric")
            print(f"DEBUG: Calling forecast_daily with lat={g['lat']}, lon={g['lon']}, units={units}")
            wx = forecast_daily(g["lat"], g["lon"], units=units)
            print(f"DEBUG: Weather forecast result keys: {list(wx.keys()) if wx else None}")
            facts["weather_by_place"][g["name"]] = {"place": g, "forecast": wx}
            facts["weather_current"] = g["name"]
            place_tz = wx.get("timezone")
            if place_tz:
                facts["today"] = today(place_tz)
            
            # Remember the user's selected place name, not just the geocoded name
            # This ensures that user selections like "2" -> "Lyon" are properly remembered
            place_to_remember = place_to_use  # Use the user's selection
            profile_update = remember_place(state, place_to_remember)
        else:
            print(f"DEBUG: geocode returned None for place: {place_to_use}")

    if plan.get("country") and place_to_use:
        # Try to get country name from the message if available
        country_name, city_name = resolve_country_and_city(state)
        country_to_lookup = country_name or place_to_use
        
        try:
            cf = country_facts(country_to_lookup)
            if cf:
                facts["country"] = cf
                country_code = country_code or cf.get("iso2") or cf.get("cca2") or cf.get("code")
                if not profile_update:
                    profile_update = remember_place(state, cf["name"])
            else:
                print(f"DEBUG: country_facts returned None for: {country_to_lookup}")
        except Exception as e:
            print(f"DEBUG: country_facts failed for {country_to_lookup}: {e}")
            # Continue without country facts rather than crashing

    if plan.get("web"):
        res = web_search(state["user_msg"], max_results=4)
        if isinstance(res, dict) and "error" not in res:
            facts["web"] = [{"title": it.get("title"), "url": it.get("url")} for it in (res.get("results") or [])[:3]]

    # daily-only time targets
    time_plan = (state.get("data") or {}).get("time_plan") or {}
    target_dates: List[str] = []
    if time_plan:
        tt = time_plan.get("target_type", "unspecified")
        if tt in ("date", "range") and (time_plan.get("iso_dates") or time_plan.get("iso_start")):
            if time_plan.get("iso_dates"):
                target_dates = time_plan["iso_dates"]
            elif time_plan.get("iso_start") and time_plan.get("iso_end"):
                d0 = date.fromisoformat(time_plan["iso_start"])
                d1 = date.fromisoformat(time_plan["iso_end"])
                days = (d1 - d0).days
                target_dates = [(d0 + timedelta(days=i)).isoformat() for i in range(max(0, days + 1))]
        elif tt in ("today", "tomorrow", "weekend"):
            base_iso = facts["today"]
            localized_weekend = weekend_for_country(country_code)
            print(f"DEBUG: country_code for weekend: {country_code}, weekend: {localized_weekend}")
            target_dates = resolve_relative_dates(tt, base_iso, weekend=localized_weekend)
            facts["weekend_days"] = localized_weekend  # e.g., (4, 5) for Fri–Sat

    if not target_dates:
        target_dates = [facts["today"]]

    facts["target_dates"] = target_dates

    merged = dict(state.get("data", {}))
    prev_wbp = (merged.get("facts") or {}).get("weather_by_place", {})
    if prev_wbp:
        facts["weather_by_place"] = {**prev_wbp, **facts["weather_by_place"]}

    merged["facts"] = deep_merge(merged.get("facts") or {}, facts)

    out: Dict[str, Any] = {"data": merged}
    out = deep_merge(out, profile_update)
    return out

def compose_answer(state: GraphState) -> Dict[str, Any]:
    facts = state.get("data", {}).get("facts", {}) or {}
    now_raw = facts.get("now", "")
    # Clean up the timestamp - just show the date
    now_clean = now_raw.split("T")[0] if "T" in now_raw else now_raw
    place_for_answer = resolve_place(state)

    facts_brief = ""
    wbp = facts.get("weather_by_place", {})
    wx_entry = wbp.get(place_for_answer) if place_for_answer else None
    target_dates = [d for d in (facts.get("target_dates") or [facts.get("today")]) if d]

    if wx_entry and target_dates:
        daily = wx_entry["forecast"]["daily"]
        dates = daily["time"]
        place = wx_entry["place"]["name"]
        parts = []
        for td in target_dates:
            if td in dates:
                idx = dates.index(td)
                tmax = daily["temperature_2m_max"][idx]
                tmin = daily["temperature_2m_min"][idx]
                pprec = daily.get("precipitation_probability_max", [None] * len(dates))[idx]
                seg = f"{td}: {tmax}°C/{tmin}°C" + (f", precip {pprec}%" if pprec is not None else "")
                parts.append(seg)
        if parts:
            if len(parts) == 1:
                facts_brief += f"Weather for {place}: {parts[0]}. "
            else:
                facts_brief += f"Weather for {place}: " + "; ".join(parts) + ". "
    elif facts.get("weather_current") and wbp.get(facts["weather_current"]):
        other = facts["weather_current"]
        if place_for_answer and other.lower() != place_for_answer.lower():
            facts_brief += f"(Note: latest weather fetched is for {other}; say 'check weather for {place_for_answer}' to refresh.) "

    if "country" in facts:
        cf = facts["country"]
        facts_brief += f"Country: {cf['name']}, capital {cf['capital']}, currency {', '.join(cf['currencies'])}. "

    # Add location context for distance-based queries
    profile = state.get("user_profile", {})
    location_data = profile.get("location_data")
    if location_data:
        facts_brief += f"User location: {location_data['location_string']} (lat: {location_data['latitude']:.2f}, lon: {location_data['longitude']:.2f}). "

    # Only include web sources for non-weather responses
    intent = state.get("intent", "")
    if "web" in facts and intent != "weather":
        links = "; ".join(f"[{i+1}] {link['title']}" for i, link in enumerate(facts["web"]))
        facts_brief += f"Web sources: {links}. "

    recent_msgs = state.get("history", [])
    recent = recent_msgs[-4:] if len(recent_msgs) >= 4 else recent_msgs
    recent_pairs = "\n".join(f"{m['role']}: {m['content']}" for m in recent if m.get("content"))
    summary = state.get("summary", "(none)")

    user_prompt = COMPOSE_TMPL.format(
        system=SYSTEM_PROMPT,
        facts=facts_brief or "none",
        summary=summary,
        recent=recent_pairs or "(none)",
        facts_policy=STRICT_FACTS_POLICY,
        checklist=REASONING_CHECKLIST,
        user_msg=state["user_msg"],
        now=now_clean or "now",
    )

    res = chat_completion_structured(
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        schema=ComposeOut,
        temperature=0.0,
    )

    draft = res.answer
    
    # Only critique if the draft is very long or has low confidence
    # For weather responses with facts, trust the LLM more
    critique_needed = len(draft) > 800 or (res.confidence < 0.5)
    
    return {"draft": draft, "critique_needed": critique_needed}

def critique(state: GraphState) -> Dict[str, Any]:
    facts = state.get("data", {}).get("facts", {})
    facts_brief = "yes" if facts else "no"
    draft = state.get("draft", "")
    
    # If the draft is already good and concise, don't critique
    if len(draft) < 200 and "weather" in draft.lower() and "°C" in draft:
        return {"critique_notes": "OK"}
    
    notes = chat_completion_simple(
        [
            {"role": "system", "content": "Be terse."},
            {"role": "user", "content": f"""Act as a strict reviewer. Check the draft against:
1) factuality vs. fetched data, 2) answers the question,
3) concise and structured, 4) no hallucinated specifics, 5) next step included.
Return either "OK" or "ISSUES: <short bullet list of fixes>"

Draft:
{draft}

Facts present? {facts_brief}"""},
        ],
        temperature=0.0,
    )
    return {"critique_notes": notes}

def revise(state: GraphState) -> Dict[str, Any]:
    notes = state.get("critique_notes", "")
    draft = state.get("draft", "")
    
    if notes.startswith("ISSUES"):
        improved = chat_completion_simple(
            [
                {"role": "system", "content": "Revise per critique, preserve facts."},
                {"role": "user", "content": f"Draft:\n{draft}\n\nCritique:\n{notes}\n\nRewrite cleanly."},
            ],
            temperature=0.2,
        )
        return {"final": improved}
    else:
        # If critique says OK, use the draft as final
        return {"final": draft}

def update_summary(state: GraphState) -> Dict[str, Any]:
    prev = state.get("summary", "")
    user = state["user_msg"]
    assistant = state.get("final") or state.get("draft", "")
    if not assistant:
        return {"summary": prev}
    summary_text = chat_completion_simple(
        [
            {"role": "system", "content": "You are a careful note-taker."},
            {"role": "user", "content": SUMMARY_TMPL.format(prev=prev or "(none)", user=user, assistant=assistant)},
        ],
        temperature=0.1,
    )
    return {"summary": summary_text.strip()}