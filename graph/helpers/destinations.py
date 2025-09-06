from __future__ import annotations
from typing import Dict, Any, Optional, Tuple
import re

def _push_destination(profile: Dict[str, Any], name: str) -> Dict[str, Any]:
    if not name:
        return profile
    lst = [p for p in profile.get("destinations", []) if p.lower() != name.lower()]
    lst.append(name)
    profile["destinations"] = lst
    profile["active_destination"] = name
    profile["destination"] = name  # legacy mirror
    return profile

def remember_place(state: Dict[str, Any], place_name: Optional[str]) -> dict:
    """Write normalized place into user_profile MRU list."""
    if not place_name:
        return {}
    profile = (state.get("user_profile") or {}).copy()
    profile = _push_destination(profile, place_name)
    return {"user_profile": profile}

def _extract_country_and_city(msg: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Extract country and city from messages like:
    - "traveling to bulgaria to sofia" -> ("Bulgaria", "Sofia")
    - "going to paris" -> (None, "Paris")
    - "visiting italy" -> ("Italy", None)
    """
    msg_lower = msg.lower()
    
    # Pattern: "to [country] to [city]" or "in [country] to [city]"
    country_city_pattern = r'\b(?:to|in|visiting|traveling to)\s+([a-z\s]+?)\s+(?:to|in)\s+([a-z\s]+?)(?:\s|$)'
    match = re.search(country_city_pattern, msg_lower)
    if match:
        country = match.group(1).strip().title()
        city = match.group(2).strip().title()
        return country, city
    
    # Pattern: "to [city], [country]" or "in [city], [country]"
    city_country_pattern = r'\b(?:to|in|visiting)\s+([a-z\s]+?),\s*([a-z\s]+?)(?:\s|$)'
    match = re.search(city_country_pattern, msg_lower)
    if match:
        city = match.group(1).strip().title()
        country = match.group(2).strip().title()
        return country, city
    
    # Pattern: just a single place (could be country or city)
    single_place_pattern = r'\b(?:to|in|visiting|traveling to)\s+([a-z\s]+?)(?:\s|$)'
    match = re.search(single_place_pattern, msg_lower)
    if match:
        place = match.group(1).strip().title()
        # Heuristic: if it's a common country name, treat as country
        common_countries = {'italy', 'france', 'spain', 'germany', 'bulgaria', 'romania', 'greece', 'turkey', 'israel', 'jordan', 'egypt'}
        if place.lower() in common_countries:
            return place, None
        else:
            return None, place
    
    return None, None

def _extract_place_from_message(msg: str) -> Optional[str]:
    # very naive fallback: first Capitalized token
    toks = [t.strip(",.!?") for t in msg.split() if t[:1].isupper()]
    return toks[0] if toks else None

def _resolve_pronoun_to_place(msg_lower: str, profile: Dict[str, Any]) -> Optional[str]:
    lst = profile.get("destinations", [])
    if not lst:
        return None
    if "previous" in msg_lower or "last one" in msg_lower or "the one before" in msg_lower:
        return lst[-2] if len(lst) >= 2 else lst[-1]
    if "first" in msg_lower or "original" in msg_lower:
        return lst[0]
    if "last" in msg_lower and "last one" not in msg_lower:
        return lst[-1]
    return None

def _resolve_place_selection(msg: str, place_candidates: List[str]) -> Optional[str]:
    """Resolve user selection like '2' to actual place name from candidates."""
    msg = msg.strip()
    
    # Check if it's a number selection
    if msg.isdigit():
        idx = int(msg) - 1  # Convert to 0-based index
        if 0 <= idx < len(place_candidates):
            return place_candidates[idx]
    
    # Check if it's an exact match with candidates
    for candidate in place_candidates:
        if msg.lower() == candidate.lower():
            return candidate
    
    return None

def resolve_place(state: Dict[str, Any]) -> Optional[str]:
    """
    Priority:
      1) LLM-resolved (data.resolved_place)
      2) place selection from candidates (e.g., "2" -> "Lyon")
      3) explicit name in message (fallback heuristic)
      4) pronoun/ordinal mapping via history (fallback)
      5) active destination
    """
    data = state.get("data") or {}
    if data.get("resolved_place"):
        return data["resolved_place"]
    
    # Check for place selection from candidates
    place_candidates = data.get("place_candidates", [])
    if place_candidates:
        msg = state.get("user_msg", "")
        selected_place = _resolve_place_selection(msg, place_candidates)
        if selected_place:
            return selected_place

    msg = state.get("user_msg", "")
    
    # Check if message contains pronouns like "there", "here", "this place"
    msg_lower = msg.lower()
    if any(pronoun in msg_lower for pronoun in ["there", "here", "this place", "that place"]):
        # Use the active destination from profile
        profile = (state.get("user_profile") or {})
        active = profile.get("active_destination") or profile.get("destination")
        if active:
            return active
    
    explicit = _extract_place_from_message(msg)
    if explicit:
        return explicit

    profile = (state.get("user_profile") or {})
    pronoun = _resolve_pronoun_to_place(msg.lower(), profile)
    if pronoun:
        return pronoun

    return profile.get("active_destination") or profile.get("destination")

def resolve_country_and_city(state: Dict[str, Any]) -> Tuple[Optional[str], Optional[str]]:
    """
    Extract country and city from the user message.
    Returns (country, city) tuple.
    """
    msg = state.get("user_msg", "")
    return _extract_country_and_city(msg)
