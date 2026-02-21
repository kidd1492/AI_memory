from langchain_ollama import OllamaEmbeddings
from memory import RAGDatabase
import numpy as np

db = RAGDatabase()
embedding_model = OllamaEmbeddings(model='mxbai-embed-large:335m')


def embed_messages(content):
    embedding = embedding_model.embed_query(content)
    embedding_array = np.array(embedding, dtype=np.float32)
    return embedding_array


def check_memory(embedding):
    retrieved = db.search_similar(embedding, top_k=3)
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