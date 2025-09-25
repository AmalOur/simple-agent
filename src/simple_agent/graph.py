"""LangGraph workflow that connects to Qwen LLM and embeddings.""" 

import os 
from langchain_openai import ChatOpenAI, OpenAIEmbeddings 
from langgraph.graph import END, StateGraph 
from dataclasses import dataclass
from typing import Optional

@dataclass
class AgentState(dict): 
    """State for the simple LangGraph agent.""" 
    input: str 
    output: Optional[str] = None 
    
    def get_llm(): 
        """Lazily initialize the chat model from environment variables.""" 
        api_key = os.getenv("LLM_API_KEY") 
        if not api_key: 
            raise RuntimeError("Missing environment variable: LLM_API_KEY") 
        
        return ChatOpenAI( 
            model="Qwen/Qwen3-30B-A3B", 
            base_url="https://inference-instance-qwen3-30b-ust2hkbr.ai.gcore.dev/v1", api_key=api_key, 
        ) 
        
    def get_embeddings(): 
        """Lazily initialize the embeddings model from environment variables.""" 
        api_key = os.getenv("EMB_API_KEY") 
        if not api_key: 
            raise RuntimeError("Missing environment variable: EMB_API_KEY") 
            
        return OpenAIEmbeddings( 
            model="Alibaba-NLP/gte-Qwen2-1.5B-instruct", 
            base_url="https://inference-instance-gte-qwen2-ust2hkbr.ai.gcore.dev/v1/embeddings", api_key=api_key, 
        ) 
        
    def call_llm(state: AgentState): 
        """Call the LLM with the user input and return its response.""" 
        llm = get_llm() response = llm.invoke(state["input"]) 
        return {"output": response.content} 
        
# ✅ Build workflow 
builder = StateGraph(AgentState) 
builder.add_node("agent", call_llm) 
builder.set_entry_point("agent") 
builder.add_edge("agent", END) 
graph = builder.compile() 
graph.name = "SimpleAgent" 

__all__ = ["graph"]