import os

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI

load_dotenv()

# The agent and the reviewer come from different companies (OpenAI vs Alibaba), so they have different
# blind spots. Override them with the OPENAI_MODEL / GROQ_REVIEW_MODEL settings (Groq retires models often).
AGENT_MODEL = "gpt-5.4-mini"  # fast and cheap: the agent makes several calls per message
REVIEW_MODEL = "qwen/qwen3.8-27b"


def make_agent_model() -> ChatOpenAI:
    # reasoning_effort="none": GPT-5.4 only accepts function tools on chat completions without reasoning.
    return ChatOpenAI(model=os.getenv("OPENAI_MODEL") or AGENT_MODEL, reasoning_effort="none", max_retries=2)


def make_review_model() -> ChatGroq:
    # A verdict is short; max_tokens keeps each review under Qwen's free-tier cap of 1,000 output tokens a minute.
    return ChatGroq(model=os.getenv("GROQ_REVIEW_MODEL") or REVIEW_MODEL, temperature=0, max_tokens=400, max_retries=2)
