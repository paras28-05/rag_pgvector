
import numpy as np
import time
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
        self.table_name = "faq_embeddings"

    def get_embedding(self, text: str) -> List[float]:
        """Generate embeddings using Gemini."""
        for attempt in range(3):
            try:
                response = embed_content(
                    model="models/text-embedding-004",
                    content=text,
                    task_type="retrieval_document",
                )
                embedding = response.get("embedding", [])
                return embedding
            except Exception as e:
                logging.error(f"Embedding generation failed: {e}")
                time.sleep(2 ** attempt)
        return []

    def normalize_embedding(self, embedding: List[float]) -> List[float]:
        """Normalize the embedding vector."""
        if not embedding:
            logging.error("Skipping normalization due to empty embedding.")
            return []
        norm = np.linalg.norm(embedding)
        return (np.array(embedding) / norm).tolist() if norm > 0 else embedding

    def insert_data(self, content: str, metadata: dict):
        """Insert data with normalized embedding into the database."""
        embedding = self.get_embedding(content)
        if not embedding:
            logging.error("Skipping insert due to missing embedding.")
            return

        normalized_embedding = self.normalize_embedding(embedding)

        insert_query = f"""
        INSERT INTO {self.table_name} (id, metadata, content, embedding)
        VALUES (%s, %s::jsonb, %s, %s::vector);
        """
        try:
            record_id = str(uuid_from_time(datetime.now()))
            self.cursor.execute(insert_query, (record_id, json.dumps(metadata), content, normalized_embedding))
            self.conn.commit()
            logging.info(f"Inserted record with ID: {record_id}")
        except Exception as e:
            logging.error(f"Error inserting data: {e}")
            self.conn.rollback()

    def upsert(self, records_df: pd.DataFrame):
        """Upsert records into the database using batch insertion."""
        records = []
        for _, row in records_df.iterrows():
            record_id = row["id"]
            metadata = json.dumps(row["metadata"])
            content = row["contents"]
            embedding = np.array(row["embedding"]).tolist()

            records.append((record_id, metadata, content, embedding))

        upsert_query = f"""
        INSERT INTO {self.table_name} (id, metadata, content, embedding)
        VALUES %s
        ON CONFLICT (id) DO UPDATE 
        SET metadata = EXCLUDED.metadata,
            content = EXCLUDED.content,
            embedding = EXCLUDED.embedding;
        """
        try:
            execute_values(self.cursor, upsert_query, records, template="(%s, %s::jsonb, %s, %s::vector)")
            self.conn.commit()
            logging.info(f"Successfully upserted {len(records)} records.")
        except Exception as e:
            logging.error(f"Error upserting data: {e}")
            self.conn.rollback()

    def search(self, query_text: str, limit: int = 3) -> pd.DataFrame:
        """Search for similar content based on query text using `pgvector`."""
        query_embedding = self.get_embedding(query_text)
        if not query_embedding:
            logging.error("Skipping search due to missing embedding.")
            return pd.DataFrame()

        query_embedding = self.normalize_embedding(query_embedding)

        search_query = f"""
        SELECT id, new_metadata, content, 1 - (embedding <=> %s::vector) AS similarity
        FROM {self.table_name}
        ORDER BY similarity DESC
        LIMIT %s;
        """
        try:
            self.cursor.execute(search_query, (query_embedding, limit))
            results = self.cursor.fetchall()
            return pd.DataFrame(results, columns=["id", "new_metadata", "content", "similarity"])
        except Exception as e:
            logging.error(f"Error executing search: {e}")
            return pd.DataFrame()
        

"""
import os
os.environ['TQDM_DISABLE'] = '1'
import logging
logging.getLogger('grpc').setLevel(logging.ERROR)  # Suppress gRPC warnings
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import json
import logging

class VectorStore:
    def __init__(self, csv_file_path: str, embedding_model: str = 'all-mpnet-base-v2'):
        self.df = pd.read_csv(csv_file_path)
        self.model = SentenceTransformer(embedding_model)
        # Generate embeddings if they don't exist in the CSV
        if 'embedding' not in self.df.columns:
            self.df['embedding'] = self.df['content'].apply(self._generate_embedding)
            self.df['embedding'] = self.df['embedding'].apply(lambda x: x.tolist()) # Convert numpy array to list
            self.df.to_csv(csv_file_path, index=False) # Save the dataframe with embeddings to the CSV file
            #logging.info("Embeddings generated and saved to CSV.").setLevel(logging.ERROR) # Or logging.WARNING, logging.CRITICAL
        else:
            # If embeddings are in the CSV, convert them from string to list
            self.df['embedding'] = self.df['embedding'].apply(json.loads)
            #logging.info("Embeddings loaded from CSV.")

    def _generate_embedding(self, text: str) -> np.ndarray:
        return self.model.encode(text)

    def search(self, query_text: str, limit: int = 50) -> pd.DataFrame:
        query_embedding = self._generate_embedding(query_text)
        # Normalize query embedding
        query_embedding = query_embedding / np.linalg.norm(query_embedding)
        
        def calculate_similarity(row):
            embedding = np.array(row['embedding'])
            embedding = embedding / np.linalg.norm(embedding)
            return np.dot(query_embedding, embedding)

        self.df['similarity'] = self.df.apply(calculate_similarity, axis=1)
        results = self.df.nlargest(limit, 'similarity').copy()
        results['category']=results['metadata'].apply(lambda x: x.get('category','Unkown') if isinstance(x, dict) else 'Unknown')
        #return results[['id','metadata','content','similarity']] # Return id, metadata, content and similarity
        return results[['id', 'category', 'content', 'similarity']]
# Example usage (in your main script or a test):
# vector_store = VectorStore("your_data.csv")
# results = vector_store.search("your query")
# print(results)
"""

import os
import time
import logging
import json
import numpy as np
import psycopg2
from config.settings import get_settings
from typing import List, Dict, Any
from google.generativeai import embed_content  # Gemini API

# Disable tqdm progress bars
os.environ['TQDM_DISABLE'] = '1'

# Suppress gRPC warnings
logging.getLogger('grpc').setLevel(logging.ERROR)


class VectorStore:
    def __init__(self):
        """
        Initializes the connection to PostgreSQL."""
        self.settings = get_settings()
        self.conn = psycopg2.connect(self.settings.database.service_url)
        self.cursor = self.conn.cursor()
        self.table_name = "faq_embeddings"  # Ensure this matches your DB schema

    def get_embedding(self, text: str) -> List[float]:
        """Generate embeddings using Gemini API."""
        for attempt in range(3):  # Retry up to 3 times in case of API failure
            try:
                response = embed_content(
                    model="models/text-embedding-004",
                    content=text,
                    task_type="retrieval_document",
                )
                embedding = response.get("embedding", [])
                if embedding:
                    return embedding
            except Exception as e:
                logging.error(f"Embedding generation failed (Attempt {attempt+1}): {e}")
                time.sleep(2 ** attempt)  # Exponential backoff
        return []

    def normalize_embedding(self, embedding: List[float]) -> List[float]:
        """Normalize the embedding vector."""
        if not embedding:
            logging.error("Skipping normalization due to empty embedding.")
            return []
        norm = np.linalg.norm(embedding)
        return (np.array(embedding) / norm).tolist() if norm > 0 else embedding

    def search(self, query_text: str, limit: int = 50) -> Dict[str, Any]:
        """
        Searches for similar text embeddings using cosine similarity and returns JSON output.
        """
        query_embedding = self.get_embedding(query_text)
        if not query_embedding:
            logging.error("Failed to generate query embedding.")
            return {"status": "error", "message": "Failed to generate query embedding"}
        print(query_embedding)
        search_query = f"""
        SELECT id, new_metadata, content, 1 - (embedding <=> %s::vector) AS similarity
        FROM {self.table_name}
        ORDER BY similarity DESC
        LIMIT %s;
        """

        try:
            self.cursor.execute(search_query, (query_embedding, limit))
            results = self.cursor.fetchall()

            if not results:
                return {"status": "success", "results": []}

            # Convert results into JSON format
            formatted_results = []
            for row in results:
                metadata = row[1]  # new_metadata column
                if isinstance(metadata, str):  # Convert JSON string to dict
                    try:
                        metadata = json.loads(metadata)
                    except json.JSONDecodeError:
                        metadata = {}  # Default to empty dict if parsing fails
                elif metadata is None:
                    metadata = {}

                formatted_results.append({
                    "id": row[0],
                    "category": metadata.get("category", "Unknown"),
                    "content": row[2],
                    "similarity": round(row[3], 4)  # Round similarity for better readability
                })

            return {"status": "success", "results": formatted_results}

        except Exception as e:
            logging.error(f"Error executing search: {e}")
            return {"status": "error", "message": str(e)}

    def close(self):
        """Closes the database connection."""
        self.cursor.close()
        self.conn.close()
