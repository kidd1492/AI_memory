from langchain.agents import create_agent
from langchain.agents.middleware import SummarizationMiddleware
from langchain.messages import HumanMessage
from langchain_ollama import ChatOllama
from langgraph.checkpoint.sqlite import SqliteSaver
import sqlite3
from token_log import log_turn_metrics


db_path = "example.db"
conn = sqlite3.connect(db_path, check_same_thread=False)
memory = SqliteSaver(conn)

model = ChatOllama(
    model="qwen2.5:3b"
)


agent = create_agent(
    model= model,
    checkpointer=memory,
    middleware=[
        SummarizationMiddleware(
            model= model,
            trigger=("messages", 6),
            keep=("messages", 2),
        ),
    ],
)

config = {"configurable": {"thread_id": "1"}}
while True:
        user_input = input("Enter: ")
        if user_input.lower() in ["exit", "quit", "q"]:
            break
        inputs = {"messages": [HumanMessage(content=user_input)]}
        result = agent.invoke(inputs, config)
        print(f"{result['messages'][-1].content}\n")
        usage = result['messages'][-1].usage_metadata
        log_turn_metrics(turn_id=0, input_text=user_input, usage=usage, file="token_metrics_mw.jsonl")