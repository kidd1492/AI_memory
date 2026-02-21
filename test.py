from langchain_ollama import ChatOllama
from langgraph.graph import START, END, StateGraph, MessagesState
from langchain_core.tools import tool
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.messages import HumanMessage, SystemMessage
from token_log import log_turn_metrics
from memory_service import *

@tool
def search_memory(query:str):
    """Return a top k simularity search from past messeges in the conversation."""
    print("tool called", "\n")
    embed_query = embed_messages(query)
    results = check_memory(embed_query)
    return results


model = ChatOllama(model="qwen2.5:3b") #.bind_tools(tools=[search_memory])


class AgentState(MessagesState):
    retrieved_memory: str | None


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
    memory_context = state["retrieved_memory"]

    system_prompt = SystemMessage(
        content=(
            "Use the following conversation to inform your response:\n"
            + memory_context
        )
    )

    messages = state["messages"]
    full_prompt = [system_prompt] + messages

    response = model.invoke(full_prompt)
    print(f"\n{response.content}\n")
    
    # Logging
    usage = response.usage_metadata
    log_turn_metrics(
        turn_id=db.count(),
        input_text=messages[-1].content,
        memory_context=memory_context,
        response_text=response.content,
        usage=usage
    )

    embedding_array = embed_messages(response.content)
    store_response(response, embedding_array)

    return {"messages": [response]}


tool_node = ToolNode(tools=[search_memory])

graph = StateGraph(AgentState)

graph.add_node("human_node", human_node)
graph.add_node("chat_node", chat_node)
#graph.add_node("tools", tool_node)

graph.add_edge(START, "human_node")
graph.add_edge("human_node", "chat_node")
#graph.add_conditional_edges("chat_node", tools_condition, {"tools": "tools", "__end__": END})
#graph.add_edge("tools", "chat_node")
graph.add_edge("chat_node", END)

agent = graph.compile()


if __name__ == "__main__":
    while True:
        user_input = input("Enter: ")
        if user_input.lower() in ["exit", "quit", "q"]:
            db.close()
            break
        inputs = {"messages": [HumanMessage(content=user_input)]}
        agent.invoke(inputs)
