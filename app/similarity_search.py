from fastapi import FastAPI, HTTPException
import logging
from database.vector_store import VectorStore
from services.synthesizer import Synthesizer

# Initialize FastAPI
app = FastAPI()

# Configure logging
logging.basicConfig(filename="rag_system.log", level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Initialize VectorStore
vector_store = VectorStore()

@app.get("/")
def root():
    return {"message": "RAG-based system is running!"}

@app.get("/query")
def query_faq(question: str):
    """Fetch the most relevant FAQ answer for the given question."""
    try:
        logging.info(f"📥 User Question: {question}")

        # Search for similar vectors in the database
        results = vector_store.search_similar_vectors(question)

        # Generate a structured JSON response
        response = Synthesizer.generate_response(question=question, context=results)

        return response
    except Exception as e:
        logging.error(f" Error processing request: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal Server Error")

# Run FastAPI using: uvicorn main:app --reload
