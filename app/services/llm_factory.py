import os
import logging
import google.generativeai as genai
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(filename="rag_system.log", level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

class LLMFactory:
    def __init__(self):
        try:
            genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
            logging.info("✅ Gemini API initialized successfully.")
        except Exception as e:
            logging.error(f"❌ Error initializing Gemini API: {str(e)}")
            raise

    def generate_query(self, user_question):
        """Generate an SQL query dynamically from the user question."""
        try:
            prompt = f"Generate an SQL query to retrieve the most relevant FAQ from the `faq_embeddings` table based on this question: {user_question}"
            response = genai.generate_text(model="gemini-2.0-flash", prompt=prompt)
            sql_query = response.text.strip()

            logging.info(f"✅ Generated SQL Query: {sql_query}")
            return sql_query
        except Exception as e:
            logging.error(f"❌ Error generating SQL query: {str(e)}")
            return None
