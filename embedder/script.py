from dotenv import load_dotenv
import os
import json
from qdrant_client import QdrantClient
from qdrant_client.http import models
from fastembed import SparseTextEmbedding
import voyageai
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s',datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger(__name__)

load_dotenv()

# Get API key from environment variables
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
VOYAGE_API_KEY = os.getenv("VOYAGE_API_KEY")

#Init clients
qdrant_client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
voyage_client = voyageai.Client(VOYAGE_API_KEY)
sparse_embedder = SparseTextEmbedding(model_name="prithivida/Splade_PP_en_v1")



def load_chunks(input_json_path, limit=None):
    """ Loads the articles from the json file and returns a list of articles """
    articles = []
    with open(input_json_path, "r") as f:
        count = 0
        for line in f:
            if line.strip():  # Skip empty lines
                try:
                    article = json.loads(line)
                    articles.append(article)
                    count += 1
                    if limit is not None and count >= limit:
                        break
                except json.JSONDecodeError as e:
                    logger.error(f"Error parsing line: {e}")
                    continue
    
    logger.info(f"Loaded {len(articles)} articles from {input_json_path}" + 
                (f" (limited to {limit})" if limit is not None else ""))
    return articles

def generate_dense_embeddings(articles, voyage_client):
    """ Generates dense embeddings for the articles """
    logger.info("Generating dense embeddings")
    dense_embeddings = []
    for i, article in enumerate(articles):
        logger.info(f"Generating dense embedding for article {i+1} of {len(articles)}")
        dense_embedding = voyage_client.embed(texts=[article['headline'] + " " + article['short_description'] + " " + article['link']], model="voyage-multilingual-2").embeddings[0]
        dense_embeddings.append(dense_embedding)
    logger.info("Dense embeddings generated")
    return dense_embeddings

def create_sparse_embeddings(articles, sparse_embedder):
    """ Creates sparse embeddings for the articles """
    logger.info("Creating sparse embeddings")
    sparse_embeddings = []
    for i, article in enumerate(articles):
        logger.info(f"Creating sparse embedding for article {i+1} of {len(articles)}")
        sparse_embedding = next(sparse_embedder.embed([article['short_description']]))
        sparse_embeddings.append(sparse_embedding)
    logger.info("Sparse embeddings created")
    return sparse_embeddings

def setup_qdrant_collection(qdrant_client, collection_name):
    """ Sets up the Qdrant collection """
    logger.info(f"Setting up Qdrant collection {collection_name}")
    
    # Check if collection already exists
    if qdrant_client.collection_exists(collection_name):
        logger.info(f"Collection '{collection_name}' already exists")
        return
    
    # Create collection since it doesn't exist
    logger.info(f"Creating new collection '{collection_name}'")
    qdrant_client.create_collection(
        collection_name=collection_name,
        vectors_config={
            "dense": models.VectorParams(size=1024, distance=models.Distance.COSINE, on_disk=True),
        },
        optimizers_config=models.OptimizersConfigDiff(
            default_segment_number=5,
            indexing_threshold=0
        ),
        sparse_vectors_config={
            "sparse": models.SparseVectorParams(
                
            )
        },
        #Add binary quantization to improve performance
        quantization_config=models.BinaryQuantization(
            binary=models.BinaryQuantizationConfig(always_ram=True),
        )
    )
    logger.info(f"Qdrant collection {collection_name} setup complete")




def prepare_and_upsert_points(articles, dense_embeddings, sparse_embeddings, qdrant_client, collection_name):
    """ Prepares and upserts the points to the Qdrant collection """
    logger.info("Preparing and upserting points")
    points = []
    for i, (article, dense_embedding, sparse_embedding) in enumerate(zip(articles, dense_embeddings, sparse_embeddings)):
        # Create the sparse vector object
        sparse_vector = models.SparseVector(
            values=sparse_embedding.values,
            indices=sparse_embedding.indices,
        )
        
        # Create the point with both vectors under a single "vector" parameter
        point = models.PointStruct(
            id=i,
            vector={
                "dense": dense_embedding,
                "sparse": sparse_vector
            },
            payload={
                "headline": article['headline'],
                "short_description": article['short_description'],
                "link": article['link'],
                "category": article['category']
            }
        )
        points.append(point)
    logger.info("Points prepared")
    logger.info(f"Upserting {len(points)} points to Qdrant collection {collection_name}")
    try:
        qdrant_client.upsert(
            collection_name=collection_name,
            points=points
        )
        logger.info(f"Upserted {len(points)} points to Qdrant collection {collection_name}")
    except Exception as e:
        logger.error(f"Error upserting points to Qdrant collection {collection_name}: {e}")
        raise e

def update_collection_settings(qdrant_client, collection_name):
    """ Updates the collection settings """
    logger.info(f"Updating collection settings for {collection_name}")
    qdrant_client.update_collection(
        collection_name=collection_name,
        optimizer_config=models.OptimizersConfigDiff(
            indexing_threshold=20000
        )
    )
    logger.info(f"Collection {collection_name} settings updated with indexing_threshold=20000")
        
def main():
    """ Main function to load the articles, generate embeddings, and upsert them to Qdrant """
    # Load only the first 1000 entries
    articles = load_chunks("./data/news_dataset_v3.json", limit=1000)
    
    dense_embeddings = generate_dense_embeddings(articles, voyage_client)
    sparse_embeddings = create_sparse_embeddings(articles, sparse_embedder)
    setup_qdrant_collection(qdrant_client, "articles")
    prepare_and_upsert_points(articles, dense_embeddings, sparse_embeddings, qdrant_client, "articles")
    update_collection_settings(qdrant_client, "articles")


if __name__ == "__main__":
     main()