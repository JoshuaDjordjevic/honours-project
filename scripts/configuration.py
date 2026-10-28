from pathlib import Path
from typing import Dict, List, Union

from langchain_ollama import ChatOllama
from langchain.tools import tool, BaseTool
from langchain.messages import (
    SystemMessage,
    HumanMessage,
    ToolMessage,
    AIMessage
)

from scripts.document_database_build import DocumentVectorStore


# Controlled system prompt, though does require
# slight modification depending on configuration
def build_system_message(use_tools: bool = False, use_rag: bool = False) -> SystemMessage:
    prompt = (
        'You are a helpful assistant. '
        'provide factual accurate responses, and keep response '
        'length under 100 words whenever possible.'
    )
    if use_tools:
        prompt += ' Use tools as required.'
    if use_rag:
        prompt += (' Additionally use the '
            'crystal rock group documentation '
            'search tool if required.')
    return SystemMessage(prompt)

# RAG data, only use if allowed

RAG_DOCUMENT_VECTOR_STORE = DocumentVectorStore(
    Path('./documents/CRG_Documentation.md'),
    Path('./chroma_db'),
    'crystal_rock_docs'
)

# RAG search tool

@tool
def search_crystal_rock_group_documentation(query: str) -> str:
    """Search the documentation for crystal
    rock group for a specific query. Returns
    search results in the form of additional
    context."""
    results = RAG_DOCUMENT_VECTOR_STORE.similarity_search(query)
    return '\n\n'.join(
        doc.page_content for doc in results
    )

# Tools to use if allowed

@tool
def get_date_time() -> str:
    """Get current date and time information.
    Returns the location, date, time, and timezone."""
    return '\n'.join([
        f'Location: Sydney Australia',
        'Date: 14 March 2026',
        'Time: 15:30 AEDT (UTC+11)'
    ])

@tool
def get_weather() -> str:
    """Get current local weather. Returns
    current conditions, humidity, wind, and
    tomorrow's forecast."""
    return '\n'.join([
        f'Location: Sydney Australia',
        'Current conditions: 22C, partly cloudy',
        'Humidity: 60%',
        'Wind: 15 km/h from the East',
        'Tomorrow\'s forecast: rain, with 5mm expected'
    ])

PROVIDED_TOOLS = [get_date_time, get_weather]
MAX_TOOL_ROUNDS = 4

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
            use_rag: bool = False,
            use_tools: bool = False):

        self.use_rag = use_rag
        self.use_tools = use_tools
        self.tools = []

        # Allow use of the RAG search tool if specified
        if self.use_rag:
            self.tools.append(search_crystal_rock_group_documentation)

        # Allow use of realtime tools if specified
        if self.use_tools:
            self.tools.extend(PROVIDED_TOOLS)

        self.tools_by_name = {t.name: t for t in self.tools}
        
        self.llm = ChatOllama(
            model=model_name,
            validate_model_on_init=True,
            temperature=temperature,
            seed=seed
        )

        if len(self.tools) > 0:
            self.llm = self.llm.bind_tools(self.tools)

    def invoke(
        self,
        query: str,
        max_tool_rounds: int = MAX_TOOL_ROUNDS
    ) -> AIMessage:
        messages = [
            build_system_message(self.use_tools, self.use_rag),
            HumanMessage(content=query)
        ]

        self.last_token_usage = {
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
            "model_calls": 0,
        }

        def track_usage(message: AIMessage) -> None:
            usage = message.usage_metadata or {}

            self.last_token_usage["input_tokens"] += (
                usage.get("input_tokens", 0)
            )
            self.last_token_usage["output_tokens"] += (
                usage.get("output_tokens", 0)
            )
            self.last_token_usage["total_tokens"] += (
                usage.get("total_tokens", 0)
            )
            self.last_token_usage["model_calls"] += 1

        # Initial model response
        result = self.llm.invoke(messages)
        track_usage(result)

        tool_rounds = 0

        while result.tool_calls and tool_rounds < max_tool_rounds:
            messages.append(result)
            tool_rounds += 1

            for tool_call in result.tool_calls:
                tool_name = tool_call["name"]
                tool_args = tool_call["args"]
                tool_call_id = tool_call["id"]

                tool = self.tools_by_name.get(tool_name)

                if tool is None:
                    tool_result = f"Error: Unknown tool '{tool_name}'"
                else:
                    try:
                        tool_result = tool.invoke(tool_args)
                    except Exception as exc:
                        tool_result = f"Tool error: {exc}"

                messages.append(
                    ToolMessage(
                        content=str(tool_result),
                        tool_call_id=tool_call_id
                    )
                )

            # Continue with the tool results
            result = self.llm.invoke(messages)
            track_usage(result)

        return result

if __name__ == '__main__':
    config = LLMConfiguration(use_tools=False)
    result = config.invoke('What is the weather in Sydney?')
    print(result.content)