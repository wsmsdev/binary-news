from dotenv import load_dotenv
import os
import voyageai
from qdrant_client import QdrantClient
from qdrant_client.http import models
from fastembed import SparseTextEmbedding
import logging


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger(__name__)


load_dotenv()
logger.info("Environment variables loaded")

VOYAGE_API_KEY = os.getenv("VOYAGE_API_KEY")
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")


voyage_client = voyageai.Client(api_key=VOYAGE_API_KEY)
logger.info("Voyage AI client initialized")

sparse_embedder = SparseTextEmbedding(model_name="prithivida/Splade_PP_en_v1")
logger.info("Sparse embedding model initialized")

qdrant_client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
logger.info(f"Qdrant client initialized at {QDRANT_URL}")


collection_name = "articles"

def get_dense_embedding(text):
    """Generate dense embedding for the given text using VoyageAI."""
    try:
        response = voyage_client.embed(texts=[text], model="voyage-multilingual-2")
        return response.embeddings[0]
    except Exception as e:
        logger.error(f"Error generating dense embedding: {e}")
        raise

def get_sparse_embedding(text):
    """Generate sparse embedding for the given text using SparseTextEmbedding."""
    try:
        sparse_embedding = next(sparse_embedder.embed([text]))
        return models.SparseVector(
            indices=sparse_embedding.indices.tolist(),
            values=sparse_embedding.values.tolist()
        )
    except Exception as e:
        logger.error(f"Error generating sparse embedding: {e}")
        raise

def hybrid_search(query, top_k=5):
    """Perform hybrid search using prefetch and RRF."""
    try:
        # Generate embeddings for the query
        dense_embedding = get_dense_embedding(query)
        sparse_embedding = get_sparse_embedding(query)
        logger.info(f"Generated embeddings for query: '{query}'")

        # Define prefetch queries for dense and sparse vectors
        prefetch_dense = models.Prefetch(
            query=dense_embedding,
            using="dense",
            limit=top_k * 2
        )
        prefetch_sparse = models.Prefetch(
            query=sparse_embedding,
            using="sparse",
            limit=top_k * 2
        )

        # Execute hybrid search with prefetch and RRF fusion
        logger.info("Performing hybrid search with prefetch and RRF")
        search_results = qdrant_client.query_points(
            collection_name=collection_name,
            prefetch=[prefetch_dense, prefetch_sparse],
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            limit=top_k,
            with_payload=True
        ).points

        logger.info(f"Found {len(search_results)} results from hybrid search")
        return search_results

    except Exception as e:
        logger.error(f"Error in hybrid search: {str(e)}", exc_info=True)
        raise

def display_article(article):
    """Helper function to display article information nicely."""
    print(f"Headline: {article.get('headline', 'N/A')}")
    print(f"Description: {article.get('short_description', 'N/A')}")
    print(f"Link: {article.get('link', 'N/A')}")
    
    content = article.get('content', '')
    if content:
        print("Content Preview:")
        print(content[:300] + "..." if len(content) > 300 else content)
    
    print(f"URL: {article.get('link', 'N/A')}")
    print("-" * 80)

def main():
    """Main function for interactive retrieval with hybrid search."""
    try:
        print("News Article Retrieval System")
        print("Enter your search query (or 'exit' to quit)")
        
        while True:
            query = input("\nSearch query: ")
            if query.lower() in ('exit', 'quit'):
                break
            
            if not query.strip():
                continue
                
            top_k = 5  
            
            
            search_results = hybrid_search(query, top_k=top_k)
            
            if not search_results:
                print("No results found.")
                continue
                
            print(f"\nFound {len(search_results)} relevant articles:")
            print("=" * 80)
            
            
            for i, point in enumerate(search_results, start=1):
                print(f"Result #{i} (Score: {point.score:.4f})")
                display_article(point.payload)
                
    except Exception as e:
        print(f"An error occurred: {e}")
        logger.error(f"Error in main: {str(e)}", exc_info=True)

if __name__ == "__main__":
    main()
