import numpy as np
import json
import logging
import psycopg2
import pandas as pd
from typing import List
from psycopg2.extras import execute_values
from datetime import datetime
from timescale_vector.client import uuid_from_time
from config.settings import get_settings
from google.generativeai import embed_content

class VectorStore:
    def __init__(self):
        """Initialize database connection and ensure table creation."""
        self.settings = get_settings()
        self.conn = psycopg2.connect(self.settings.database.service_url)
        self.cursor = self.conn.cursor()
        self.table_name = self.settings.vector_store.table_name
        self.create_table()

    def create_table(self):
        """Create `faq_embeddings` table if it does not exist."""
        create_table_query = f"""
        CREATE TABLE IF NOT EXISTS {self.table_name} (
            id UUID PRIMARY KEY,
            metadata JSONB,
            content TEXT NOT NULL,
            embedding vector({self.settings.vector_store.embedding_dimensions}) NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """
        try:
            self.cursor.execute(create_table_query)
            self.conn.commit()
            logging.info(f"Table '{self.table_name}' is ready.")
        except Exception as e:
            logging.error(f"Error creating table '{self.table_name}': {e}")

    def get_embedding(self, text: str) -> List[float]:
        """Generate embeddings using Gemini."""
        try:
            response = embed_content(
                model="models/text-embedding-004",
                content=text,
                task_type="retrieval_document",
            )
            embedding = response.get("embedding", [])
            logging.info(f"🔹 Generated Query Embedding: {len(embedding)} dimensions")  # Debugging print
            return embedding
        except Exception as e:
            logging.error(f"Embedding generation failed: {e}")
            return []

    def normalize_embedding(self, embedding: List[float]) -> List[float]:
        """Normalize the embedding vector."""
        norm = np.linalg.norm(embedding)
        return embedding / norm if norm > 0 else embedding
    print("ok")

    def insert_data(self, content: str, metadata: dict):
        """Insert data with normalized embedding into the database."""
        embedding = self.get_embedding(content)
        if not embedding:
            logging.error("Skipping insert due to missing embedding.")
            return
        
        normalized_embedding = self.normalize_embedding(embedding)
        normalized_embedding = normalized_embedding.tolist()

        insert_query = f"""
        INSERT INTO {self.table_name} (id, metadata, content, embedding)
        VALUES (%s, %s, %s, %s::vector);
        """
        try:
            # Generate UUID for the record
            record_id = str(uuid_from_time(datetime.now()))
            self.cursor.execute(insert_query, (record_id, metadata, content, normalized_embedding))
            self.conn.commit()
            logging.info(f"Inserted data into {self.table_name}")
        except Exception as e:
            logging.error(f"Error inserting data: {e}")

    def upsert(self, records_df: pd.DataFrame):
        """Upsert records into the database."""
        for _, row in records_df.iterrows():
            record_id = row['id']
            metadata = json.dumps(row['metadata'])
            content = row['contents']
            embedding = row['embedding']

            # Convert embedding to list
            embedding = np.array(embedding).tolist()
    
            upsert_query = f"""
            INSERT INTO {self.table_name} (id, content, embedding, metadata)
            VALUES (%s, %s, %s, %s::vector)
            """
            try:
                self.cursor.execute(upsert_query, (record_id, metadata, content, embedding))
                self.conn.commit()
                logging.info(f"Upserted record with ID {record_id}.")
            except Exception as e:
                logging.error(f"Error upserting data: {e}")

    def search(self, query_text: str, limit: int = 3) -> pd.DataFrame:
        """Search for similar content based on query text using `pgvector`."""
        query_embedding = self.get_embedding(query_text)
        if not query_embedding:
            logging.error("Skipping search due to missing embedding.")
            return pd.DataFrame()

        # Normalize the query embedding
        query_embedding = self.normalize_embedding(query_embedding)

        # Convert numpy ndarray to list
        query_embedding = query_embedding.tolist()

        search_query = f"""
        SELECT "id", "metadata", "content", 1 - (embedding <=> %s::vector) AS similarity
        FROM {self.table_name}
        ORDER BY similarity DESC
        LIMIT %s;
        """
        try:
            self.cursor.execute(search_query, (query_embedding, limit))
            results = self.cursor.fetchall()
            return pd.DataFrame(results, columns=["id", "metadata", "content", "similarity"])
        except Exception as e:
            logging.error(f"Error executing search: {e}")
            return pd.DataFrame()