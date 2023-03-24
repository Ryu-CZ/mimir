from functools import cached_property
from typing import Any

from langchain.chains.llm import LLMChain
from langchain.graphs.networkx_graph import (
    KnowledgeTriple,
    parse_triples,
    get_entities,
)
from langchain.memory import ConversationKGMemory, ChatMessageHistory
from langchain.schema import (
    BaseMemory,
    get_buffer_string,
)
from pydantic import Field

__all__ = (
    "MultiModel",
    "ConversationGraphMemory"
)


class MultiModel(BaseMemory):
    memories: list[BaseMemory] = Field(default_factory=list)
    memory_variables_: list[str] = Field(default_factory=list)
    save_only_input: bool = True

    def add(self, memory: BaseMemory) -> None:
        """A new named memory free"""
        for variable_name in memory.memory_variables:
            if variable_name in self.memory_variables_:
                raise KeyError(
                    f"{type(self).__qualname__} {variable_name!r} introducing in memory {memory} already exists")
        self.memories.append(memory)
        self.memory_variables_.extend(memory.memory_variables)

    @property
    def memory_variables(self) -> list[str]:
        """Will always return list of memory variables.

        :meta private:
        """
        return [*self.memory_variables_]

    def load_memory_variables(self, inputs: dict[str, Any]) -> dict[str, Any]:
        memorized = {}
        for memory in self.memories:
            memorized.update(memory.load_memory_variables(inputs))
        return memorized

    def save_context(self, inputs: dict[str, str], outputs: dict[str, str]) -> None:
        if self.save_only_input:
            inputs = {"input": inputs["input"]}
        for memory in self.memories:
            memory.save_context(inputs, outputs)

    def clear(self) -> None:
        for memory in self.memories:
            memory.clear()


class ChatMessageHistoryBuffer(ChatMessageHistory):
    max_len: int = 6

    def allocate_memory(self):
        if len(self.messages) == self.max_len:
            self.messages.pop(0)

    def add_ai_message(self, message: str) -> None:
        self.allocate_memory()
        super().add_ai_message(message)

    def add_user_message(self, message: str) -> None:
        self.allocate_memory()
        super().add_user_message(message)


class ConversationGraphMemory(ConversationKGMemory):
    chat_memory: ChatMessageHistoryBuffer = Field(default_factory=ChatMessageHistoryBuffer)

    class Config:
        arbitrary_types_allowed = True
        keep_untouched = (cached_property,)

    @cached_property
    def entity_chain(self):
        return LLMChain(llm=self.llm, prompt=self.entity_extraction_prompt)

    @cached_property
    def knowledge_chain(self):
        return LLMChain(llm=self.llm, prompt=self.knowledge_extraction_prompt)

    def get_current_entities(self, input_string: str) -> list[str]:
        buffer_string = get_buffer_string(
            self.chat_memory.messages[-self.k * 2:],
            human_prefix=self.human_prefix,
            ai_prefix=self.ai_prefix,
        )
        output = self.entity_chain.predict(
            history=buffer_string,
            input=input_string,
        )
        return get_entities(output)

    def get_knowledge_triplets(self, input_string: str) -> list[KnowledgeTriple]:
        buffer_string = get_buffer_string(
            self.chat_memory.messages[-self.k * 2:],
            human_prefix=self.human_prefix,
            ai_prefix=self.ai_prefix,
        )
        output = self.knowledge_chain.predict(
            history=buffer_string,
            input=input_string,
            verbose=True,
        )
        knowledge = parse_triples(output)
        return knowledge
