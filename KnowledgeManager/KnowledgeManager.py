import os
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


class KnowledgeManager:
    def __init__(self, 
                 index_path=None, 
                 embedding_model_name='paraphrase-multilingual-mpnet-base-v2'):
        """
        Initialize the KnowledgeManager.
        
        :param index_path: Path to the FAISS index file.
        :param embedding_model_name: Pre-trained embedding model name.
        """
        self.index_path = index_path or os.path.join("KnowledgeManager", "faiss_index.bin")
        self.embedding_model = SentenceTransformer(embedding_model_name)
        self.index = None

    def load_faiss_index(self):
        """
        Load the FAISS index from the specified path.
        """
        if not os.path.exists(self.index_path):
            raise FileNotFoundError(f"FAISS index file not found at {self.index_path}")
        
        self.index = faiss.read_index(self.index_path)
        print(f"FAISS index loaded from {self.index_path}.")

    def search(self, query, top_k=3, distance_threshold=0.8):
        """
        Search the FAISS index for the top results matching the query.
        
        :param query: The input query as a string.
        :param top_k: Number of top results to retrieve.
        :param distance_threshold: Threshold for filtering results by distance.
        :return: A list of indices matching the query.
        """
        # Ensure FAISS index is loaded
        if self.index is None:
            print("Loading FAISS index...")
            self.load_faiss_index()

        # Compute query embedding
        query_embedding = self.embedding_model.encode(query, 
                                                      convert_to_numpy=True, 
                                                      normalize_embeddings=True)
        query_embedding = np.array([query_embedding], dtype="float32")

        # Search the FAISS index
        distances, indices = self.index.search(query_embedding, top_k)

        # Filter results based on distance threshold
        results = [
            int(idx) for dist, idx in zip(distances[0], indices[0])
            if dist <= distance_threshold and idx != -1
        ]

        print(f"Search complete. Query: '{query}' | Results: {results}")
        return results
