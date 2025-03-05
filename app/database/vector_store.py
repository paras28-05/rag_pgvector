import psycopg2
import os
import json
import logging
import google.generativeai as genai
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(filename="rag_system.log", level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

class VectorStore:
    def __init__(self):
        """Initialize database connection and Gemini API."""
        try:
            self.conn = psycopg2.connect(
                dbname=os.getenv("DB_NAME"),
                user=os.getenv("DB_USER"),
                password=os.getenv("DB_PASSWORD"),
                host=os.getenv("DB_HOST"),
                port=os.getenv("DB_PORT"),
            )
            self.cursor = self.conn.cursor()
            logging.info("✅ Database connection established successfully.")

            # Configure Gemini API
            genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
        except Exception as e:
            logging.error(f"❌ Error initializing VectorStore: {str(e)}")
            raise

    def get_embedding(self, text):
        """Generate 768-dimensional embeddings using Gemini's `embedding-004`."""
        try:
            response = genai.embed_content(
                model="models/text-embedding-004",
                content=text,
                task_type="retrieval_document"
            )
            logging.info("✅ Successfully generated embedding for input text.")
            return response["embedding"]
        except Exception as e:
            logging.error(f"❌ Error generating embedding: {str(e)}")
            return None

    def search_similar_vectors(self, query_text):
        """Convert question to embedding and fetch the most relevant answers."""
        try:
            # Generate embedding for the input question
            user_embedding = self.get_embedding(query_text)
            if user_embedding is None:
                return {"status": "error", "message": "Failed to generate embedding"}

            # ✅ Fetch **top 3** most similar answers instead of just 1
            sql_query = """
                SELECT id, category, content
                FROM faq_embeddings
                ORDER BY embedding <=> %s::vector
                LIMIT 3;
            """
            logging.info("Executing Query: %s", sql_query)

            self.cursor.execute(sql_query, (user_embedding,))
            results = self.cursor.fetchall()

            if results:
                return [{"id": row[0], "category": row[1], "content": row[2]} for row in results]

            return {"error": "No relevant answers found"}

        except Exception as e:
            logging.error(f"Error executing search: {str(e)}")
            return {"status": "error", "message": str(e)}
