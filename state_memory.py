from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, RemoveMessage
from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.checkpoint.sqlite import SqliteSaver
import sqlite3
import json

class AgentState(MessagesState):
    summary: str


db_path = "example.db"
conn = sqlite3.connect(db_path, check_same_thread=False)
memory = SqliteSaver(conn)
model = ChatOllama(model="qwen2.5:3b")


def chat(state: AgentState):
    summary = state.get("summary", "")
    user_message = state["messages"][-1].content

    if summary:
        system_message = f"Summary of conversation earlier: {summary}"
        messages = [SystemMessage(content=system_message)] + state["messages"]
    else:
        messages = state["messages"]

    response = model.invoke(messages)
    usage = response.usage_metadata
    print("\n", response)

    log_turn_metrics(
        turn_id=len(state["messages"]),
        input_text=user_message,
        summary_context=summary,
        full_prompt=messages,  # this includes SystemMessage + HumanMessage
        response_text=response.content,
        usage=usage
    )


    return {"messages": response}


def log_turn_metrics(turn_id, input_text, summary_context, full_prompt, response_text, usage):
    log_entry = {
        "turn": turn_id,
        "input": input_text,
        "summary_context": summary_context,
        "full_prompt": [msg.dict() for msg in full_prompt],  # serialize LangChain messages
        "response": response_text,
        "input_tokens": usage.get("input_tokens"),
        "output_tokens": usage.get("output_tokens"),
        "total_tokens": usage.get("total_tokens")
    }
    with open("token_metrics_systemA.jsonl", "a") as f:
        f.write(json.dumps(log_entry) + "\n")



def summary_node(state: AgentState):
    summary = state.get("summary", "")
    if summary:
        summary_message =(
            f"This is a summary of the conversation to date: {summary}\n\n"
            "Extend the summary by taking into account the new messages above:"
        )
    else:
        summary_message = "Create a summary of the conversation above: "

    messages =  state["messages"] + [HumanMessage(content=summary_message)]
    response = model.invoke(messages)

    delete_messages = [RemoveMessage(id=m.id) for m in state["messages"][:-2]]
    return {"summary": response.content, "messages": delete_messages}


def should_summarize(state: AgentState): #-> Literal ["summarize_conversation", "end"]:
    messages =  state["messages"]
    if len(messages) >= 6:
        return "summarized"
    else:
        return "human"


graph = StateGraph(AgentState)


graph.add_node("chat", chat)
graph.add_node("summary_node", summary_node) 

graph.add_edge(START, "chat")
graph.add_conditional_edges(
    "chat", 
    should_summarize, 
    {
        "summarized" : "summary_node",
        "human" : END
    }
)
agent = graph.compile(checkpointer=memory)

config = {"configurable": {"thread_id": "1"}}
user_input = input("Enter Question: ")
inputs = {"messages": [HumanMessage(content=user_input)]}
agent.invoke(inputs, config)

