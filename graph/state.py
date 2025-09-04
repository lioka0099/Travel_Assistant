from re import A
from typing import TypedDict, List, Dict, Any, Optional

from langchain_core.runnables import history

class GraphState(TypedDict, total = False): #total = False makes it possible to do paartial updates
    history: List[Dict[str,str]] #[{"role": "user" | "assistant", "content": "what was written"}]
    user_msg: str
    intent: str # what was the purpose of the user's query (packing, sedtination, weather etc..)
    user_profile: Dict[str, Any] # slots for memory of the conversation (what destination, start and end date etc..)
    data: Dict[str, Any] # plan + facts (clock, weather, country, web)
    draft: str #draft for reflection if needed
    final: str
    critique_needed: bool
    critique_notes: Optional[str]
    offtopic_counter: int # for small talk redirection level
    summary: str