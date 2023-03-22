from typing import Any
from pydantic import Field

from langchain.schema import BaseMemory

__all__ = (
    "MultiModel",
)


class MultiModel(BaseMemory):
    memories: list[BaseMemory] = Field(default_factory=list)
    memory_variables_: list[str] = Field(default_factory=list)
    save_only_input: bool = True

    def add(self, memory: BaseMemory) -> None:
        """A new named memory free"""
        for variable_name in memory.memory_variables:
            if variable_name in self.memory_variables_:
                raise KeyError(f"{type(self).__qualname__} {variable_name!r} introducing in memory {memory} already exists")
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
