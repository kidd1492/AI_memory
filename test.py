from langchain_ollama import ChatOllama, OllamaEmbeddings
from langgraph.graph import START, END, StateGraph
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
import numpy as np
from typing import TypedDict, List
from memory import RAGDatabase
import json


model = ChatOllama(model="qwen2.5:3b")
embedding_model = OllamaEmbeddings(model='mxbai-embed-large:335m')

class AgentState(TypedDict):
    messages: HumanMessage | AIMessage


def log_turn_metrics(turn_id, input_text, memory_context, response_text, usage):
    log_entry = {
        "turn": turn_id,
        "input": input_text,
        "retrieved_memory": memory_context,
        "response": response_text,
        "input_tokens": usage.get("input_tokens"),
        "output_tokens": usage.get("output_tokens"),
        "total_tokens": usage.get("total_tokens")
    }
    with open("token_metrics.jsonl", "a") as f:
        f.write(json.dumps(log_entry) + "\n")


def chat_node(state: AgentState) -> AgentState:
    db = RAGDatabase()
    user_message = state['messages'][0]
    query_embedding = embedding_model.embed_query(user_message.content)
    query_array = np.array(query_embedding, dtype=np.float32)

    # Retrieve memory if available
    print(f"\nMemory count: {db.count()}")
    if db.count() == 0:
        memory_context = "No prior memory available."
    else:
        retrieved = db.search_similar(query_array, top_k=5)
        memory_context = "\n".join([f"{role}: {content}" for role, content in retrieved]) or "No relevant memory found."

    # Construct prompt
    system_prompt = SystemMessage(content="Use the following conversation to inform your response:\n" + memory_context)
    full_prompt = [system_prompt, user_message]
    print(f"\nfull_prompt: {full_prompt}\n")

    response = model.invoke(full_prompt)
    usage = response.usage_metadata
    print(f"\n token usage: {usage}")
    print(response.content)
    log_turn_metrics(
        turn_id=db.count(),  # You can increment this or timestamp it
        input_text=user_message.content,
        memory_context=memory_context,
        response_text=response.content,
        usage=usage
    )

    return {"messages": [response]}

def embedding_node(state: AgentState):
    db = RAGDatabase()
    message = state['messages'][0]
    embedding = embedding_model.embed_query(message.content)
    embedding_array = np.array(embedding, dtype=np.float32)
    db.add_message(
        role=message.type,
        content=message.content,
        embedding=embedding_array
    )
    #print("Stored embedding:", embedding_array)

def should_continue(state: AgentState):
    message = state['messages'][0]
    if isinstance(message, HumanMessage):
        return "chat"
    else:
        return "end"

graph = StateGraph(AgentState)
graph.add_node('chat_node', chat_node)
graph.add_node('embedding_node', embedding_node)

graph.add_edge(START, 'embedding_node')
graph.add_conditional_edges('embedding_node', should_continue, {
    'chat': 'chat_node',
    'end': END
})
graph.add_edge('chat_node', 'embedding_node')
agent = graph.compile()

user_input = input("Enter Question: ")
inputs = {"messages": [HumanMessage(content=user_input)]}
agent.invoke(inputs)
