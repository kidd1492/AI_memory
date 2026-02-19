import sqlite3, json
import numpy as np
from typing import List, Tuple

class RAGDatabase:
    def __init__(self, db_path='rag_store.db'):
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self._init_schema()

    def _init_schema(self):
        cursor = self.conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                embedding BLOB NOT NULL
            )
        """)
        self.conn.commit()


    def add_message(self, role: str, content: str, embedding: np.ndarray):
        cursor = self.conn.cursor()
        embedding_blob = embedding.tobytes()
        cursor.execute("""
            INSERT INTO messages (role, content, embedding)
            VALUES (?, ?, ?)
        """, (role, content, embedding_blob))
        self.conn.commit()


    def count(self) -> int:
        cursor = self.conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM messages")
        return cursor.fetchone()[0]


    def search_similar(self, query_embedding: np.ndarray, top_k: int = 5, threshold: float = 0.5) -> List[Tuple[str, str]]:
        cursor = self.conn.cursor()
        cursor.execute("SELECT role, content, embedding FROM messages")
        results = []
        for role, content, embedding_blob in cursor.fetchall():
            stored_embedding = np.frombuffer(embedding_blob, dtype=np.float32)
            similarity = np.dot(query_embedding, stored_embedding) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(stored_embedding)
            )
            if similarity >= threshold:
                results.append((similarity, role, content))
        results.sort(reverse=True)
        print(f"\nResults similarity: {results}")
        return [(role, content) for similarity, role, content in results[:top_k]]
