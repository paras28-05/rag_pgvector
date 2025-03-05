import pandas as pd
import json
from datetime import datetime
from database.vector_store import VectorStore
from timescale_vector.client import uuid_from_time

# Initialize VectorStore
vec = VectorStore()

# Read the CSV file
df = pd.read_csv("D:\\rag_pg\\pgvectorscale-rag-solution\\data\\faq_dataset.csv", sep=";")

# Prepare data for insertion
def prepare_record(row):
    """Prepare a record for insertion into the vector store."""
    content = f"Question: {row['question']}\nAnswer: {row['answer']}"
    embedding = vec.get_embedding(content)
    
    return pd.Series({
        "id": str(uuid_from_time(datetime.now())),  # Generate UUID
        "metadata": json.dumps({
            "category": row["category"],
            "created_at": datetime.now().isoformat(),
        }),
        "content": content,
        "embedding": json.dumps(embedding)  # Convert list to JSON for CSV
    })

# Process all records
records_df = df.apply(prepare_record, axis=1)

# Save embeddings to CSV
csv_path = "D:\\rag_pg\\pgvectorscale-rag-solution\\data\\faq_embeddings.csv"
#vec.save_to_csv(records_df, csv_path)

# Insert embeddings from CSV into PostgreSQL
vec.load_from_csv(csv_path)
