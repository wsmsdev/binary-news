# Binary-News

A hybrid search news retrieval system powered by vector embeddings and binary quantization.

## About

Binary-News is an advanced news retrieval application that uses state-of-the-art vector search techniques to find relevant news articles based on user queries. The system combines the strengths of dense and sparse embeddings to provide highly accurate search results while optimizing for performance.

### Key Features

- **Hybrid Search**: Combines the semantic understanding of dense vectors with the keyword precision of sparse vectors
- **Binary Quantization**: Employs binary quantization to optimize vector storage and improve search speed
- **Streamlit Interface**: User-friendly web interface for searching and exploring news articles
- **Fast Retrieval**: Efficiently retrieves relevant articles in milliseconds
- **Multilingual Support**: Built with VoyageAI's multilingual model for cross-language search capabilities

## Technical Overview

The system works in two main stages:

1. **Embedding & Indexing**:

   - News articles are embedded using VoyageAI for dense vectors and SPLADE for sparse vectors
   - Embeddings are stored in Qdrant vector database with binary quantization
   - Collection is optimized for efficient retrieval

2. **Search & Retrieval**:
   - User queries are embedded using the same models
   - Hybrid search combines results from both vector spaces using Reciprocal Rank Fusion
   - Results are ranked by relevance and presented through the Streamlit interface

## Getting Started

### Prerequisites

- Python 3.11+
- API keys for VoyageAI and Qdrant
- Access to the news dataset

### Installation

1. Clone the repository
2. Install dependencies: `pip install -r requirements.txt`
3. Create a `.env` file with required API keys
4. Download the news dataset and place it in the project directory

### Running the Application

1. Run the embedding script: `python embedder/script.py`
2. Launch the Streamlit interface: `streamlit run main.py`

## Dataset

News dataset provided by:

1. Misra, Rishabh. "News Category Dataset." arXiv preprint arXiv:2209.11429 (2022).
2. Misra, Rishabh and Jigyasa Grover. "Sculpting Data for ML: The first act of Machine Learning." ISBN 9798585463570 (2021).

Kaggle link: https://www.kaggle.com/datasets/rishabhmisra/news-category-dataset

## License

MIT

## Acknowledgements

- VoyageAI for providing the dense embedding model
- SPLADE for sparse embeddings
- Qdrant for vector storage
- Streamlit for the web interface
