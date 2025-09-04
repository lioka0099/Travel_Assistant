import re

#Used as fallback is the llm fails to determine the intent of the user.

def hint_weather(user_msg: str) -> bool:
    weather_terms = r"(weather|rain|temperature|forecast|sunny|snow|wind)"
    date_terms = r"(today|tomorrow|weekend|\b\d{2}-\d{2}-\d{4}\b)"
    return bool(re.search(weather_terms, user_msg, re.I)) or bool(re.search(date_terms, user_msg, re.I))

def hint_country_facts(user_msg: str) -> bool:
    fact_terms = r"(currency|visa|language|timezone|plug|outlet|capital)"
    return bool(re.search(fact_terms, user_msg, re.I))

def hint_web_search(user_msg: str) -> bool:
    pat = r"(open\s+today|hours|closed|latest|news|strike|event(s)?\s+(today|tonight|this weekend)|update)"
    return bool(re.search(pat, user_msg, re.I))
