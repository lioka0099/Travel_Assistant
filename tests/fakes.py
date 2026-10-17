from langchain_core.language_models import BaseChatModel
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import RunnableLambda
from pydantic import Field


class ScriptedModel(BaseChatModel):
    """Fake chat model for tests: returns scripted replies in order and records every input.

    Script items are AIMessages (agent), Review objects (reviewer) or Exceptions (raised).
    """

    script: list
    calls: list = Field(default_factory=list)

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def _next(self, messages):
        self.calls.append(messages)
        item = self.script.pop(0)
        if isinstance(item, Exception):
            raise item
        return item

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        return ChatResult(generations=[ChatGeneration(message=self._next(messages))])

    def bind_tools(self, tools, **kwargs):
        return self

    def with_structured_output(self, schema, **kwargs):
        return RunnableLambda(self._next)
