from typing import Dict, List

from langchain_ollama import ChatOllama
from langchain.tools import tool, BaseTool
from langchain.messages import (
    SystemMessage,
    HumanMessage,
    ToolMessage,
    AIMessage
)

# Controlled system prompt
SYSTEM_PROMPT = SystemMessage(
    'You are a helpful assistant. Use tools when necessary, '
    'provide factual accurate responses, and keep response '
    'length under 100 words whenever possible.'
)

# Tools to use if allowed

@tool
def get_weather(city: str) -> str:
    """Get weather for a given city."""
    return f"Weather in {city}: Sunny, 21c"

PROVIDED_TOOLS = [get_weather]

# Class for invoking the LLM and
# allowing it access to tools or RAG

class LLMConfiguration:
    use_tools: bool
    tools: List[BaseTool]
    tools_by_name: Dict[str, BaseTool]

    def __init__(
            self,
            model_name: str = 'gemma4:e4b',
            temperature: float = 0.2,
            seed: int = 42,
            use_tools: bool = False):

        self.use_tools = use_tools
        self.tools = PROVIDED_TOOLS.copy()
        self.tools_by_name = {t.name: t for t in self.tools}
        
        self.llm = ChatOllama(
            model=model_name,
            validate_model_on_init=True,
            temperature=temperature,
            seed=seed
        )

        if self.use_tools:
            self.llm = self.llm.bind_tools(self.tools)

    def invoke(self, query: str) -> AIMessage:
        messages: List = [
            SYSTEM_PROMPT,
            HumanMessage(query)
        ]

        result = self.llm.invoke(messages)
        messages.append(result)

        if self.use_tools and isinstance(result, AIMessage) and result.tool_calls:
            for tool_call in result.tool_calls:
                tool_result = self.tools_by_name[tool_call['name']].invoke(
                    tool_call['args']
                )

                messages.append(
                    ToolMessage(
                        content=str(tool_result),
                        tool_call_id=tool_call['id']
                    )
                )

            # Second call: model receives tool result and responds
            result = self.llm.invoke(messages)

        return result

if __name__ == '__main__':
    config = LLMConfiguration(use_tools=False)
    result = config.invoke('What is the weather in Sydney?')
    print(result.content)