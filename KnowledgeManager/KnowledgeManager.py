import os
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from multiprocessing import Lock

class KnowledgeManager:
    # Class-level attributes
    _faiss_index = None
    _index_path = None
    _lock = Lock()  # Multiprocessing lock to ensure thread-safety

    def __init__(self, 
                 index_path=None, 
                 embedding_model_name='paraphrase-multilingual-mpnet-base-v2'):
        """
        Initialize the KnowledgeManager.
        
        :param index_path: Path to the FAISS index file.
        :param embedding_model_name: Pre-trained embedding model name.
        """
        if KnowledgeManager._index_path is None:
            KnowledgeManager._index_path = index_path or os.path.join("KnowledgeManager", "faiss_index.bin")
        
        self.embedding_model = SentenceTransformer(embedding_model_name)
        self._ensure_faiss_index_loaded()

    @classmethod
    def _ensure_faiss_index_loaded(cls):
        """
        Ensure the FAISS index is loaded across workers.
        """
        with cls._lock:  # Ensure only one worker loads the index at a time
            if cls._faiss_index is None:
                if not os.path.exists(cls._index_path):
                    raise FileNotFoundError(f"FAISS index file not found at {cls._index_path}")
                
                cls._faiss_index = faiss.read_index(cls._index_path)
                print(f"FAISS index loaded from {cls._index_path}.")

    def search(self, query, top_k=3, distance_threshold=0.8):
        """
        Search the FAISS index for the top results matching the query.
        
        :param query: The input query as a string.
        :param top_k: Number of top results to retrieve.
        :param distance_threshold: Threshold for filtering results by distance.
        :return: A list of indices matching the query.
        """
        # Ensure FAISS index is loaded
        self._ensure_faiss_index_loaded()

        # Compute query embedding
        query_embedding = self.embedding_model.encode(query, 
                                                      convert_to_numpy=True, 
                                                      normalize_embeddings=True)
        query_embedding = np.array([query_embedding], dtype="float32")

        # Search the FAISS index
        distances, indices = KnowledgeManager._faiss_index.search(query_embedding, top_k)

        # Filter results based on distance threshold
        results = [
            int(idx) for dist, idx in zip(distances[0], indices[0])
            if dist <= distance_threshold and idx != -1
        ]

        print(f"Search complete. Query: '{query}' | Results: {results}")
        return results
