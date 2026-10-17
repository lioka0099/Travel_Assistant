import os

from dotenv import load_dotenv
from langchain_groq import ChatGroq

load_dotenv()

# Groq retires models often; override these with the GROQ_MODEL / GROQ_REVIEW_MODEL settings.
AGENT_MODEL = "openai/gpt-oss-20b"  # fast and cheap: the agent makes several calls per message
REVIEW_MODEL = "qwen/qwen3.8-27b"  # a different model family, so it has different blind spots


def make_agent_model() -> ChatGroq:
    return ChatGroq(model=os.getenv("GROQ_MODEL") or AGENT_MODEL, temperature=0.2, max_retries=2)


def make_review_model() -> ChatGroq:
    return ChatGroq(model=os.getenv("GROQ_REVIEW_MODEL") or REVIEW_MODEL, temperature=0, max_retries=2)
