from langchain_ollama import ChatOllama, OllamaEmbeddings
from langgraph.graph import START, END, StateGraph
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
import numpy as np
from typing import TypedDict, List
from memory import RAGDatabase
import json


db = RAGDatabase()
model = ChatOllama(model="qwen2.5:3b")
embedding_model = OllamaEmbeddings(model='mxbai-embed-large:335m')


class AgentState(TypedDict):
    messages: List[HumanMessage | AIMessage]
    retrieved_memory: str | None
    last_embedding: np.ndarray | None


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


def chat_node(state: AgentState):
    """Generate the model response using retrieved memory."""
    user_message = state["messages"][0]
    memory_context = state["retrieved_memory"]

    system_prompt = SystemMessage(
        content="Use the following conversation to inform your response:\n" + memory_context
    )

    full_prompt = [system_prompt, user_message]
    response = model.invoke(full_prompt)

    print(f"\n{response.content}\n")

    usage = response.usage_metadata
    log_turn_metrics(
        turn_id=db.count(),
        input_text=user_message.content,
        memory_context=memory_context,
        response_text=response.content,
        usage=usage
    )

    return {
        "messages": [response],
        "retrieved_memory": memory_context,
        "last_embedding": None
    }


def store_response_node(state: AgentState):
    """Embed and store the AI response."""
    message = state["messages"][0]
    embedding = embedding_model.embed_query(message.content)
    embedding_array = np.array(embedding, dtype=np.float32)

    db.add_message(
        role=message.type,
        content=message.content,
        embedding=embedding_array
    )

    return state


graph = StateGraph(AgentState)

graph.add_node("chat_node", chat_node)
graph.add_node("store_response_node", store_response_node)

graph.add_edge(START, "chat_node")
graph.add_edge("chat_node", "store_response_node")
graph.add_edge("store_response_node", END)

agent = graph.compile()


user_input = input("Enter Question: ")
inputs = {"messages": [HumanMessage(content=user_input)]}
embedding = embedding_model.embed_query(user_input)
embedding_array = np.array(embedding, dtype=np.float32)

if db.count() == 0:
        memory_context = "No prior memory available."
else:
    retrieved = db.search_similar(embedding_array, top_k=5)
    memory_context = "\n".join([f"{role}: {content}" for role, content in retrieved]) or \
                        "No relevant memory found."
    
db.add_message(
    role='human',
    content=user_input,
    embedding=embedding_array
    )


response = agent.invoke({
    "messages": [HumanMessage(content=user_input)],
    "retrieved_memory": memory_context,
})

#.invoke(inputs)
