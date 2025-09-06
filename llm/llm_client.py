import os
from dotenv import load_dotenv
from typing import List, Dict, Optional, Type, TypeVar, Literal
from pydantic import BaseModel,Field, ConfigDict, AliasChoices
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage,HumanMessage,AIMessage
import json

load_dotenv()
T = TypeVar("T", bound=BaseModel)

def _to_lc_message(m:Dict):
    role,content = m['role'],m['content']
    if role == "system":
        return SystemMessage(content=content)
    elif role == "user":
        return HumanMessage(content=content)
    return AIMessage(content=content)

def _chat(model: str | None = None, temperature: float = 0.2) -> ChatGroq:
    model_name = model or os.getenv("GROQ_MODEL") or "llama-3.1-8b-instant"
    return ChatGroq(
        model_name=model_name,
        groq_api_key=os.environ["GROQ_API_KEY"],
        temperature=temperature,
        max_retries=2,   # mild retry for transient 5xx/validate hiccups
    )

def chat_completion_simple(messages: List[Dict], model: str | None = None, temperature: float = 0.2) -> str:
    llm = _chat(model=model, temperature=temperature)
    res = llm.invoke([_to_lc_message(m) for m in messages])
    return res.content

def chat_completion_structured(
    messages: List[Dict],
    schema: Type[T],
    model: Optional[str] = None,
    temperature: float = 0.2,
) -> T:
    """Return a validated Pydantic object of type `schema`. Robust to Groq JSON-mode failures."""
    llm = _chat(model=model, temperature=temperature)

    # Always include an explicit JSON hint to satisfy Groq's requirement
    base_msgs = (
        [{"role": "system", "content": "You must respond with a JSON object. (This message intentionally includes the word JSON.)"}]
        + messages
    )

    # Attempt 1: JSON mode (server-enforced)
    try:
        llm_struct = llm.with_structured_output(schema, method="json_mode")
        return llm_struct.invoke([_to_lc_message(m) for m in base_msgs])
    except Exception as e1:
        # Attempt 2: plain call + local parse into the schema
        try:
            res = llm.invoke([_to_lc_message(m) for m in base_msgs])
            obj = json.loads(res.content)
            return schema.model_validate(obj)
        except Exception:
            # Attempt 3: one strict retry, zero temperature
            llm_strict = _chat(model=model, temperature=0.0)
            strict_msgs = (
                [{"role": "system",
                  "content": "STRICT JSON: Return EXACTLY one JSON object that matches the schema keys/types. No prose, no backticks. (JSON)"}]
                + base_msgs
            )
            res2 = llm_strict.invoke([_to_lc_message(m) for m in strict_msgs])
            obj2 = json.loads(res2.content)
            return schema.model_validate(obj2)
    
# -- Schemas --

class ComposeOut(BaseModel):
    # Ignore extra keys (e.g., "details"), and allow validation aliases
    model_config = ConfigDict(extra="ignore", populate_by_name=True)

    # Accept answer under several common names (the left-most wins if multiple present)
    answer: str = Field(
        ...,
        description="Final user-facing answer; may include newlines and bullets",
        validation_alias=AliasChoices("answer", "summary", "text", "content"),
    )
    confidence: float = Field(..., ge=0.0, le=1.0, description="Self-rated confidence 0..1")

class ToolPlan(BaseModel):
    need_weather: bool = Field(False, description="Fetch weather/forcast?")
    need_country: bool = Field(False, description="Fetch country facts?")
    need_web: bool = Field(False, description="Use web search for fresh data?")
    place_hint: Optional[str] = Field(None, description = "If a place is implied, name it.")
    rationale: str = Field(...,description="One short reason for choices.")

class TimePlan(BaseModel):
    target_type: Literal["unspecified", "today", "tomorrow", "weekend", "date", "range"] = "unspecified"
    iso_dates: Optional[List[str]] = None   # explicit date from the user, if any
    iso_start: Optional[str] = None         # for explicit ranges
    iso_end: Optional[str] = None
    rationale: str = Field(..., description="Brief reason for the selection")

class PlacePlan(BaseModel):
    resolved_place: Optional[str] = None
    resolution: Literal["explicit","implicit_previous","implicit_first","implicit_last","none"] = "none"
    ambiguous: bool = False
    alternatives: Optional[List[str]] = None
    rationale: str = Field(..., description="≤1 line why you chose this")