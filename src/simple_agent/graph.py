"""LangGraph workflow that connects to Qwen LLM and embeddings."""

import os
from typing import TypedDict

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langgraph.graph import END, StateGraph


class AgentState(TypedDict):
    """User-facing state for the simple agent."""
    input: str  # required user-provided input


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


def get_embeddings() -> OpenAIEmbeddings:
    """Lazily initialize the embeddings model from environment variables."""
    api_key = os.getenv("EMB_API_KEY")
    if not api_key:
        raise RuntimeError("Missing environment variable: EMB_API_KEY")

    return OpenAIEmbeddings(
        model="Alibaba-NLP/gte-Qwen2-1.5B-instruct",
        base_url="https://inference-instance-gte-qwen2-ust2hkbr.ai.gcore.dev/v1/embeddings",
        api_key=api_key,
    )


def call_llm(state: AgentState) -> dict:
    """Call the LLM with the user input and return its response."""
    llm = get_llm()
    response = llm.invoke(state["input"])
    # ✅ output is added dynamically, not part of the input schema
    return {"output": response.content}


# ✅ Build workflow
builder = StateGraph(AgentState)
builder.add_node("agent", call_llm)
builder.set_entry_point("agent")
builder.add_edge("agent", END)

graph = builder.compile()
graph.name = "SimpleAgent"

__all__ = ["graph"]
