from langchain_ollama import ChatOllama, OllamaEmbeddings
from langgraph.graph import START, END, StateGraph
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage
import numpy as np
from typing import TypedDict, List
from memory import RAGDatabase
import json


@tool
def get_last_message_tool() -> str:
    """Return the last stored message content from memory."""
    last = db.get_last_message()
    if last is None:
        return "No messages stored yet."
    return last



db = RAGDatabase()
model = ChatOllama(model="qwen2.5:3b").bind_tools(tools=[get_last_message_tool])
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

TOOLS = {
    "get_last_message_tool": get_last_message_tool,
}

def tool_node(state: AgentState):
    last = state["ai_message"][0]

    if not hasattr(last, "tool_calls") or not last.tool_calls:
        return state

    tool_call = last.tool_calls[0]
    tool_name = tool_call["name"]
    tool_args = tool_call["args"]

    # Look up the correct tool dynamically
    tool_fn = TOOLS.get(tool_name)
    if tool_fn is None:
        result = f"Error: unknown tool '{tool_name}'"
    else:
        result = tool_fn.invoke(tool_args)

    tool_msg = ToolMessage(
        content=str(result),
        tool_call_id=tool_call["id"]
    )

    state["tool_message"] = [tool_msg]
    return state



def human_node(state: AgentState):
    user_message = state["messages"][0]
    user_text = user_message.content

    embedding_array = embed_messages(user_text)

    if db.count() == 0:
        memory_context = "No prior memory available."
    else:
        memory_context = check_memory(embedding_array)

    store_response(user_message, embedding_array)

    state["retrieved_memory"] = memory_context
    return state


def chat_node(state: AgentState):
    user_message = state["messages"][0]
    memory_context = state["retrieved_memory"]

    system_prompt = SystemMessage(
        content="Use the following conversation to inform your response:\n" + memory_context + "for general questions just answer. if needed tool avaliable get_last_message_tool for more context"
    )

    full_prompt = [system_prompt, user_message]

    if state.get("tool_message"):
        full_prompt.append(state["tool_message"][0])

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

    state["ai_message"] = [response]
    return state


def needs_tool(state: AgentState) -> bool:
    ai_list = state.get("ai_message", [])
    if not ai_list:
        return False
    ai_msg = ai_list[0]
    return hasattr(ai_msg, "tool_calls") and bool(ai_msg.tool_calls)


# -------------------------
# GRAPH DEFINITION
# -------------------------
graph = StateGraph(AgentState)

graph.add_node("human_node", human_node)
graph.add_node("chat_node", chat_node)
graph.add_node("tool_node", tool_node)

graph.add_edge(START, "human_node")
graph.add_edge("human_node", "chat_node")

graph.add_conditional_edges(
    "chat_node",
    needs_tool,
    {
        True: "tool_node",
        False: END,
    }
)

graph.add_edge("tool_node", "chat_node")

agent = graph.compile()


if __name__ == "__main__":
    while True:
        user_input = input("Enter: ")
        if user_input.lower() in ["exit", "quit", "q"]:
            break
        inputs = {"messages": [HumanMessage(content=user_input)]}
        agent.invoke(inputs)
