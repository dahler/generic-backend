import os
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import psycopg2
from urllib.parse import urlparse
from config.config import db_url



# db_url = os.getenv("APP_DB")
# if not db_url:
#     raise ValueError("Environment variable APP_DB is not set.")

# parsed_url = urlparse(db_url)
# POSTGRES_CONFIG = {
#     'host': parsed_url.hostname,
#     'port': parsed_url.port,
#     'user': parsed_url.username,
#     'password': parsed_url.password,
#     'database': parsed_url.path.lstrip('/')
# }
# print(POSTGRES_CONFIG)
# PostgreSQL Database Configuration
# POSTGRES_CONFIG = {
#     'host': 'localhost',
#     'port': 5432,
#     'user': 'postgres',
#     'password': 'Password',
#     'database': 'KnowledgeDB'
# }

# FAISS Index File Path
FAISS_INDEX_PATH = os.path.join("KnowledgeManager", "faiss_index.bin")

# Load Embedding Model
embedding_model = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2')


class KnowledgeManager:
    def __init__(self, index_path=FAISS_INDEX_PATH):
        """Initialize the KnowledgeManager."""
        self.index = None
        self.index_path = index_path
        self.load_faiss_index()

    def load_faiss_index(self):
        """Load the FAISS index from the specified path."""
        if os.path.exists(self.index_path):
            self.index = faiss.read_index(self.index_path)
            print("FAISS index loaded.")
        else:
            raise FileNotFoundError(f"FAISS index file not found at {self.index_path}")

    def search(self, query, top_k=3, distance_threshold=0.8):
        """
        Search the FAISS index for the top results matching the query.
        
        :param query: The input query as a string.
        :param top_k: Number of top results to retrieve.
        :param distance_threshold: Threshold for filtering results by distance.
        :return: A list of results with content, metadata, and distance.
        """
        if self.index is None:
            raise ValueError("FAISS index is not loaded. Please load the index before searching.")

        # Compute query embedding
        query_embedding = embedding_model.encode(query, convert_to_numpy=True)
        query_embedding = np.array([query_embedding], dtype="float32")

        # Search the FAISS index
        distances, indices = self.index.search(query_embedding, top_k)
        print("Distances:", indices)

        results = []
        if indices is not None and distances is not None:
            for idx in indices[0]:
                print(int(idx))
                results.append(idx)
                 
            # connection = psycopg2.connect(**POSTGRES_CONFIG)
            # cursor = connection.cursor()

            # for dist, idx in zip(distances[0], indices[0]):
            #     if idx == -1:  # Skip invalid indices
            #         continue
            #     if dist > distance_threshold:  # Include only results within the threshold
            #         cursor.execute("SELECT content, metadata FROM law_documents WHERE id = %s;", (int(idx),))
            #         result = cursor.fetchone()
            #         if result:
            #             results.append({"content": result[0], "metadata": result[1], "distance": str(dist)})

            # cursor.close()
            # connection.close()

        return results
