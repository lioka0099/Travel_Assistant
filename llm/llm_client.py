import os
from pickle import LIST
from dotenv import Optional, load_dotenv
from typing import List,Dict
from pydantic import BaseModel,Feild, Field
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage,HumanMessage,AIMessage

load_dotenv()

def _to_lc_message(m:Dict):
    role,content = m['role'],m['content']
    if role == "system":
        return SystemMessage(content=content)
    elif role == "user":
        return HumanMessage(content=content)
    return AIMessage(content=content)

def _chat(model: str | None = None, temperature: float = 0.2) -> ChatGroq:
    return ChatGroq(
        model_name = model or os.getenv("GROQ_MODEL", ""),
        qroq_api_key = os.environ["GROQ_API_KEY"],
        temperature=temperature
    )

# -- Structured output --
class ComposeOut(BaseModel):
    answer: str  = Feild(..., description="Final user facing answer")
    confidence: float = Field(..., ge=0.0, le=1.0, description="self rated confidence 0..1")

class ToolPlan(BaseModel):
    need_weather: bool = Field(False, description="Fetch weather/forcast?")
    need_country: bool = Field(False, description="Fetch country facts?")
    need_web: bool = Field(False, description="Use web search for fresh data?")
    place_hint: Optional[str] = Feild(None, description = "If a place is implied, name it.")
    rationale: str = Field(...,description="One short reason for choices.")

def chat_completion_simple(messages: List[Dict], model: str | None = None, temperature: float = 0.2) -> str:
    llm = _chat(model=model, temperature=temperature)
    res = llm.invoke([_to_lc_message(m) for m in messages])
    return res.content

def chat_completion_structured(messages: List[Dict], schema, model: Optional[str] = None, temperature: float = 0.2):
    """Return a validated Pydantic object of type `schema`."""
    llm = _chat(model=model, temperature=temperature)
    llm_struct = llm.with_structured_output(schema)
    return llm_struct.invoke([_to_lc_message(m) for m in messages])
    