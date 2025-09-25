import os
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

# Load secrets from environment (LangGraph or .env)
LLM_API_KEY = os.environ["LLM_API_KEY"]
EMB_API_KEY = os.environ["EMB_API_KEY"]

# Configure the LLM (Chat model)
llm = ChatOpenAI(
    model="Qwen/Qwen3-30B-A3B",
    base_url="https://inference-instance-qwen3-30b-ust2hkbr.ai.gcore.dev/v1",
    api_key=LLM_API_KEY,
)

# Configure the Embedding model (not used in this simple workflow yet)
embeddings = OpenAIEmbeddings(
    model="Alibaba-NLP/gte-Qwen2-1.5B-instruct",
    base_url="https://inference-instance-gte-qwen2-ust2hkbr.ai.gcore.dev/v1/embeddings",
    api_key=EMB_API_KEY,
)

# Define simple state
class AgentState(dict):
    input: str
    output: str

# Node: call the LLM
def call_llm(state: AgentState):
    response = llm.invoke(state["input"])
    return {"output": response.content}

# Build workflow
graph = StateGraph(AgentState)
graph.add_node("agent", call_llm)
graph.set_entry_point("agent")
graph.add_edge("agent", END)