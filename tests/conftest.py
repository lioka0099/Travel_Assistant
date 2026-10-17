import os

# Keep test runs offline: llm/llm_client.py loads a local .env, which may turn on LangSmith tracing.
# load_dotenv never overrides variables that are already set, and these take precedence over the others.
os.environ["LANGSMITH_TRACING_V2"] = "false"
os.environ["LANGSMITH_TRACING"] = "false"
