from typing import Dict,Any

def deep_merge(a: Dict[str,Any], b: Dict[str,Any]) -> Dict[str,Any]:
    """
    Return a new dict where nested dicts are merged (b overrides a).
    Non-dict values get replaced. Safe for LangGraph's shallow state merge.
    """
    out = dict(a or {})
    for k, v in (b or {}).items():
        if isinstance(v,dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k],v)
        else:
            out[k] = v
    return out