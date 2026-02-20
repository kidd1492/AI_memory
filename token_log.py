import json

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