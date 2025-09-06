from langchain_core.prompts import PromptTemplate

SYSTEM_PROMPT = """You are a concise, friendly travel assistant.
Rules:
- Be helpful and practical.
- Lead with a 1-sentence summary, then bullets (≤6).
- Use external data only when provided; do NOT invent hours/prices.
- Ask at most one clarification question only when necessary.
"""

STRICT_JSON_FOOTER = """
Output policy (STRICT):
- You MUST return a single JSON object.
- Use EXACT keys and value types as specified.
- Do NOT add extra fields, comments, prose, or backticks.
- Use null for unknown optional fields.
- Remember: you are returning JSON only (no text before or after).
"""

STRICT_FACTS_POLICY = """
Facts policy (MANDATORY):
- If external data is provided, you MUST use it and present it.
- Do NOT claim you lack access to data that is already provided in [Context].
- Do NOT ask the user to check other websites for the same data you already have.
"""

ROUTER_PROMPT = """Classify the user's message into exactly one intent:
- destinations, packing, attractions, logistics, smalltalk, weather

Rules:
1) If the user_msg contains weather/forecast/temperature/"today"/"tomorrow"/"weekend"/"tonight" etc. ⇒ prefer "weather".
2) If the user_msg contains pack/packing/clothes/what to wear ⇒ "packing".
3) If the user_msg asks about things to do/see/museums/attractions ⇒ "attractions".
4) If the user_msg asks about transport/visa/currency/hours/open/closed/tickets ⇒ "logistics".
5) If the message is a short acknowledgement (e.g., "ok", "thanks", "thank you", "got it", "perfect") ⇒ "smalltalk".
6) If user_msg is a short follow-up (≤ 5 tokens) AND prior_intent is "weather" AND NOT an acknowledgement ⇒ keep "weather".
7) If user_msg is a short follow-up (≤ 5 tokens) AND prior_intent is not "smalltalk" AND NOT an acknowledgement ⇒ keep prior_intent unless rule (1) picks "weather".
8) If has_travel_context is true, do NOT choose "smalltalk".
9) Only choose "smalltalk" if the message is clearly chit-chat and none of the above rules apply.

- Return ONLY the intent word.
User: {user_msg}
"""

REASONING_CHECKLIST = """Plan:
1) Identify intent & constraints (dates, destination, style).
2) Decide if live data is needed (weather? country facts? web?).
3) Outline ≤6 crisp bullets + a 1-line TL;DR.
4) Include any live facts explicitly with 'As of <now>'.
5) Offer a sensible next step or question if needed.
"""

COMPOSE_TMPL = PromptTemplate.from_template(
    """{system}

[Task]
Using the conversation and any fetched data, answer the user's latest message.

[Context]
- External data: {facts}
- Summary: {summary}
- Recent: {recent}

{facts_policy}

[Reasoning steps]
{checklist}

[Output style]
- If weather-by-date data is available, output:
  1) A one-line TL;DR.
  2) Then one bullet per date: "- YYYY-MM-DD: X°/Y° (precip P%)".
- Otherwise, reply in 2–3 crisp sentences.
- **NEVER** add sources, citations, or "Sources:" lines to weather responses.
- **NEVER** write placeholders like "[Insert source(s) ...]" or "check a reliable source".
- Do NOT include prefaces like "Here is", "Here's a revised version", "Draft:", "Summary:", or any explanation.
- Output ONLY the final answer text the user should see.

[Response contract — RETURN JSON ONLY]
Return a single JSON object with EXACTLY these keys and types:
{{
  "answer": string,      // the full user-facing answer (may include newlines and bullets)
  "confidence": number   // 0.0..1.0
}}

Do not add any other keys. Do not wrap in code fences. No prose outside JSON.

User message: "{user_msg}" """
)


SMALLTALK_REDIRECT_PROMPT = """You received a smalltalk message from the user:
"{user_msg}"

Reply warmly in 1–2 short sentences (max ~40 words).
Then pivot to travel by asking this exact question (verbatim):
"{question}"

Important:
- Do NOT state or imply any facts about destinations, weather, prices, or opening hours.
- Do NOT answer the travel request here; only pivot back to travel planning.
Keep it concise and friendly."""

SUMMARY_TMPL = PromptTemplate.from_template(
    """Update the running conversation summary.

Previous summary (may be empty):
{prev}

New exchange:
User: {user}
Assistant: {assistant}

Write a concise 3-5 line summary focused on durable facts (destination, dates, preferences, decisions).
Do NOT include word-for-word quotes; keep it compact and factual."""
)

PLANNER_SYS = f"""You decide which data tools to call for a travel assistant.

Return JSON with EXACT keys and types:
{{
  "need_weather": boolean,
  "need_country": boolean,
  "need_web": boolean,
  "place_hint": string | null,
  "rationale": string
}}

Policy:
- If the user asks about weather/temperature/forecast/conditions OR intent is 'packing' with a known place -> need_weather=true.
- If the user asks about currency/visa/language/timezone/plug -> need_country=true.
- If the user asks about "open today/hours/this weekend/latest/strike" etc. -> consider need_web=true.
- Keep rationale to ≤1 line.
- Never hallucinate.

Example:
{{
  "need_weather": true,
  "need_country": false,
  "need_web": false,
  "place_hint": "Holon",
  "rationale": "User asked for weather in Holon."
}}

{STRICT_JSON_FOOTER}
"""

TIME_PLANNER_SYS = f"""You are a time-intent normalizer for a travel assistant.

Your ONLY job is to determine WHEN the user is asking about. Do NOT provide attraction information or travel advice.

Return JSON with EXACT keys and types:
{{
  "target_type": "unspecified" | "today" | "tomorrow" | "weekend" | "date" | "range",
  "iso_dates": string[] | null,   // list of YYYY-MM-DD if explicit single/multi-date given
  "iso_start": string | null,     // YYYY-MM-DD if explicit range start
  "iso_end": string | null,       // YYYY-MM-DD if explicit range end
  "rationale": string
}}

Notes:
- Do NOT guess dates. If user is relative: prefer "today"/"tomorrow"/"weekend".
- Parts of day (tonight/evening/morning/afternoon) without a date ⇒ map to TODAY (destination timezone).
- “next weekend” ⇒ use target_type="weekend". (Resolution to specific dates happens elsewhere.)
- If nothing time-specific, use "unspecified".

Examples:
- "what can I do there?" → {{"target_type": "unspecified", "iso_dates": null, "iso_start": null, "iso_end": null, "rationale": "No specific time mentioned"}}
- "what's the weather this weekend?" → {{"target_type": "weekend", "iso_dates": null, "iso_start": null, "iso_end": null, "rationale": "User asked for weekend"}}

{STRICT_JSON_FOOTER}
"""

PLACE_RESOLVER_SYS = f"""You resolve the destination/place referenced in a user's message.

Inputs you will receive:
- message: the latest user message (string)
- active_destination: the current active destination from profile (string or null)
- destinations: an ordered list of previous destinations (array of strings)

Resolution rules:
- If the user explicitly names a place, set resolution="explicit" and put the normalized name in resolved_place.
- If the user implies a prior place (e.g., "previous", "same place", "continue"), map to the correct item in 'destinations' and set resolution accordingly:
  - "implicit_previous" → the most recently used destination
  - "implicit_first" → the first destination in history
  - "implicit_last" → the last destination in history (if different from previous)
- If multiple plausible places are present and you cannot choose one confidently, set ambiguous=true and list up to 3 alternatives in 'alternatives'. In that case, resolved_place should be null and resolution="none".
- Do NOT invent places.

Return JSON with EXACT keys and types:
{{
  "resolved_place": string | null,     // normalized place to use or null
  "resolution": "explicit" | "implicit_previous" | "implicit_first" | "implicit_last" | "none",
  "ambiguous": boolean,
  "alternatives": string[],            // up to 3; [] if none
  "rationale": string                  // ≤1 short line, why you chose this
}}

Examples (valid):
{{
  "resolved_place": "Holon",
  "resolution": "explicit",
  "ambiguous": false,
  "alternatives": [],
  "rationale": "User said 'Holon' explicitly."
}}

{{
  "resolved_place": null,
  "resolution": "none",
  "ambiguous": true,
  "alternatives": ["Holon", "Hod HaSharon"],
  "rationale": "Two similar city names mentioned; unclear."
}}

{STRICT_JSON_FOOTER}
"""

