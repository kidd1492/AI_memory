from langchain_ollama import ChatOllama, OllamaEmbeddings
from langgraph.graph import START, END, StateGraph
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage
import numpy as np
from typing import TypedDict, List
from memory import RAGDatabase
import json


db = RAGDatabase()
model = ChatOllama(model="qwen2.5:3b")
embedding_model = OllamaEmbeddings(model='mxbai-embed-large:335m')


class AgentState(TypedDict):
    messages: List[HumanMessage]
    ai_message: List[AIMessage]
    tool_message: List[ToolMessage]
    retrieved_memory: str | None


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


def embed_messages(content):
    embedding = embedding_model.embed_query(content)
    embedding_array = np.array(embedding, dtype=np.float32)
    return embedding_array


def check_memory(embedding):
    retrieved = db.search_similar(embedding, top_k=5)
    memory_context = "\n".join(
        [f"{role}: {content}" for role, content in retrieved]
    ) or "No relevant memory found."

    return memory_context


def store_response(message, embedding_array):
    text = message.content

    db.add_message(
        role=message.type,
        content=text,
        embedding=embedding_array
    )

    return


def human_node(state: AgentState):
    user_message = state["messages"][0]          # HumanMessage object
    user_text = user_message.content             # Extract text

    embedding_array = embed_messages(user_text)

    # Retrieve memory
    if db.count() == 0:
        memory_context = "No prior memory available."
    else:
        memory_context = check_memory(embedding_array)

    # Store human message in DB
    store_response(user_message, embedding_array)

    # Update state
    state["retrieved_memory"] = memory_context
    return state



def chat_node(state: AgentState):
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

    embedding_array = embed_messages(response.content)
    store_response(response, embedding_array)

    # Replace messages with the AI response
    state["ai_message"] = [response]
    return state


# -------------------------
# GRAPH DEFINITION
# -------------------------
graph = StateGraph(AgentState)

graph.add_node("human_node", human_node)
graph.add_node("chat_node", chat_node)


graph.add_edge(START, "human_node")
graph.add_edge("human_node", "chat_node")
graph.add_edge("chat_node", END)

agent = graph.compile()


if __name__ == "__main__":
    while True:
        user_input = input("Enter: ")
        if user_input.lower() in ["exit", "quit", "q"]:
            break
        inputs = {"messages": [HumanMessage(content=user_input)]}
        agent.invoke(inputs)
