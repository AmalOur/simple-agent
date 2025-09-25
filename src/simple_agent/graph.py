"""LangGraph workflow that connects to Qwen LLM and embeddings."""

import os
from typing import TypedDict

from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph


class AgentState(TypedDict):
    """State schema: only user-provided input is required."""
    input: str


def get_llm() -> ChatOpenAI:
    """Lazily initialize the chat model from environment variables."""
    api_key = os.getenv("LLM_API_KEY")
    if not api_key:
        raise RuntimeError("Missing environment variable: LLM_API_KEY")

    return ChatOpenAI(
        model="Qwen/Qwen3-30B-A3B",
        base_url="https://inference-instance-qwen3-30b-ust2hkbr.ai.gcore.dev/v1",
        api_key=api_key,
    )


def call_llm(state: AgentState) -> dict:
    """Call the LLM with the user input and return its response."""
    llm = get_llm()
    response = llm.invoke(state["input"])
    return {
        "output": response.content  # ✅ only return output
    }


# ✅ Build workflow
builder = StateGraph(AgentState)
builder.add_node("agent", call_llm)
builder.set_entry_point("agent")
builder.add_edge("agent", END)

graph = builder.compile()
graph.name = "SimpleAgent"

__all__ = ["graph"]
