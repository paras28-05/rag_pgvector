
# 🚀 Building a High-Performance RAG Solution with Pgvectorscale and Python


![GitHub stars](https://img.shields.io/github/stars/paras28-05/rag_pgvector)
![GitHub issues](https://img.shields.io/github/issues/paras28-05/rag_pgvector)
![GitHub forks](https://img.shields.io/github/forks/paras28-05/rag_pgvector)
![GitHub license](https://img.shields.io/github/license/paras28-05/rag_pgvector)

## 📚 Project Description
This tutorial will guide you through setting up and using `pgvectorscale`  and Python, leveraging Gemini's powerful  model for embeddings. You'll learn to build a cutting-edge RAG (Retrieval-Augmented Generation) solution, combining advanced retrieval techniques (including hybrid search) with intelligent answer generation based on the retrieved context. Perfect for AI engineers looking to enhance their projects with state-of-the-art vector search and generation capabilities with the power of PostgreSQL.

## 📖 Table of Contents
- [Installation Guide](#installation-guide)
- [Features](#features)
- [Configuration & Environment Variables](#configuration--environment-variables)
- [Contributing](#contributing)
- [Acknowledgments & Credits](#acknowledgments--credits)

## 🛠️ Installation Guide
Follow these steps to install and set up the project:

1. Clone the repository:
   ```bash
   git clone https://github.com/paras28-05/rag_pgvector.git
   cd rag_pgvector
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## ✨ Features
- **High-Performance Vector Search**: Utilizes PostgreSQL with `pgvectorscale` for efficient vector search capabilities.
- **Seamless Integration**: Combines relational and vector data within a single database.
- **Advanced Indexing Techniques**: Includes DiskANN-inspired indexing for faster searches.
- **Cosine Similarity**: Measures the cosine of the angle between vectors for accurate similarity search.

## ⚙️ Configuration & Environment Variables
To configure the project, you need to set the following environment variables:

- `GEMINI_API_KEY`: Your Gemini AI API key.
- `POSTGRES_URL`= postgresql+psycopg2://`user`:`password`@`host`:`port`/`database_name`
- `DB_HOST`: Database host, default is `localhost`.
- `DB_PORT`: Database port, default is `5432`.
- `DB_USER`: Database user, default is `postgres`.
- `DB_PASSWORD`: Database password, default is `password`.
- `DB_NAME`: Database name, default is `postgres`.

## 🤝 Contributing
We welcome contributions! Please follow these guidelines:

1. Fork the repository.
2. Create a new branch (`git checkout -b feature-branch`).
3. Make your changes.
4. Commit your changes (`git commit -m 'Add feature'`).
5. Push to the branch (`git push origin feature-branch`).
6. Create a new Pull Request.


## 🙏 Acknowledgments & Credits
- [paras28-05](https://github.com/paras28-05)

## Pgvectorscale Documentation

For more information about using PostgreSQL as a vector database in AI applications with Timescale, check out these resources:

- [GitHub Repository: pgvectorscale](https://github.com/timescale/pgvectorscale)
- [Blog Post: PostgreSQL and Pgvector: Now Faster Than Pinecone, 75% Cheaper, and 100% Open Source](https://www.timescale.com/blog/pgvector-is-now-as-fast-as-pinecone-at-75-less-cost/)
- [Blog Post: RAG Is More Than Just Vector Search](https://www.timescale.com/blog/rag-is-more-than-just-vector-search/)
- [Blog Post: A Python Library for Using PostgreSQL as a Vector Database in AI Applications](https://www.timescale.com/blog/a-python-library-for-using-postgresql-as-a-vector-database-in-ai-applications/)

## Why PostgreSQL?

Using PostgreSQL with pgvectorscale as your vector database offers several key advantages over dedicated vector databases:

- PostgreSQL is a robust, open-source database with a rich ecosystem of tools, drivers, and connectors. This ensures transparency, community support, and continuous improvements.
- By using PostgreSQL, you can manage both your relational and vector data within a single database. This reduces operational complexity, as there's no need to maintain and synchronize multiple databases.
- Pgvectorscale enhances pgvector with faster search capabilities, higher recall, and efficient time-based filtering. It leverages advanced indexing techniques, such as the DiskANN-inspired index, to significantly speed up Approximate Nearest Neighbor (ANN) searches.

Pgvectorscale Vector builds on top of [pgvector](https://github.com/pgvector/pgvector), offering improved performance and additional features, making PostgreSQL a powerful and versatile choice for AI applications.

## Using ANN search indexes to speed up queries

Timescale Vector offers indexing options to accelerate similarity queries, particularly beneficial for large vector datasets (10k+ vectors):

1. Supported indexes:
   - timescale_vector_index (default): A DiskANN-inspired graph index
   - pgvector's HNSW: Hierarchical Navigable Small World graph index
   - pgvector's IVFFLAT: Inverted file index

2. The DiskANN-inspired index is Timescale's latest offering, providing improved performance. Refer to the [Timescale Vector explainer blog](https://www.timescale.com/blog/pgvector-is-now-as-fast-as-pinecone-at-75-less-cost/) for detailed information and benchmarks.

For optimal query performance, creating an index on the embedding column is recommended, especially for large vector datasets.

## Cosine Similarity in Vector Search

### What is Cosine Similarity?

Cosine similarity measures the cosine of the angle between two vectors in a multi-dimensional space. It's a measure of orientation rather than magnitude.

- Range: -1 to 1 (for normalized vectors, which is typical in text embeddings)
- 1: Vectors point in the same direction (most similar)
- 0: Vectors are orthogonal (unrelated)
- -1: Vectors point in opposite directions (most dissimilar)

### Cosine Distance

In pgvector, the `<=>` operator computes cosine distance, which is 1 - cosine similarity.

- Range: 0 to 2
- 0: Identical vectors (most similar)
- 1: Orthogonal vectors
- 2: Opposite vectors (most dissimilar)

### Interpreting Results

When you get results from similarity_search:

- Lower distance values indicate higher similarity.
- A distance of 0 would mean exact match (rarely happens with embeddings).
- Distances closer to 0 indicate high similarity.
- Distances around 1 suggest little to no similarity.
- Distances approaching 2 indicate opposite meanings (rare in practice).
```
