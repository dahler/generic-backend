import os
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import asyncio

class KnowledgeManager:
    _faiss_indices = {}
    _index_paths = {}
    _locks = {}

    def __init__(self, index_name, index_path=None, embedding_model_name='paraphrase-multilingual-mpnet-base-v2'):
        self.index_name = index_name
        if index_name not in KnowledgeManager._index_paths:
            KnowledgeManager._index_paths[index_name] = index_path or os.path.join("KnowledgeManager", f"{index_name}.bin")
        if index_name not in KnowledgeManager._locks:
            KnowledgeManager._locks[index_name] = asyncio.Lock()
        
        self.embedding_model = SentenceTransformer(embedding_model_name)

    async def _ensure_faiss_index_loaded(self):
        """Asynchronously load the FAISS index if it hasn't been loaded."""
        async with KnowledgeManager._locks[self.index_name]:
            if self.index_name not in KnowledgeManager._faiss_indices:
                index_path = KnowledgeManager._index_paths[self.index_name]

                if not os.path.exists(index_path):
                    raise FileNotFoundError(f"FAISS index file not found at {index_path}")

                KnowledgeManager._faiss_indices[self.index_name] = await asyncio.to_thread(faiss.read_index, index_path)
                print(f"FAISS index '{self.index_name}' loaded from {index_path}.")

    async def search(self, query, top_k=5, distance_threshold=8):
        """Asynchronous search using FAISS."""
        await self._ensure_faiss_index_loaded()
        index = KnowledgeManager._faiss_indices[self.index_name]

        query_embedding = await asyncio.to_thread(
            self.embedding_model.encode, query, convert_to_numpy=True, normalize_embeddings=True
        )
        query_embedding = np.array([query_embedding], dtype="float32")

        distances, indices = await asyncio.to_thread(index.search, query_embedding, top_k)

        # results = [int(idx) for dist, idx in zip(distances[0], indices[0]) if dist <= distance_threshold and idx != -1]

        # print(f"Search complete. Index: '{self.index_name}' | Query: '{query}' | Results: {results}")
        # return results

        # Keep results within the distance threshold
        faiss_results = [
            (int(idx), float(dist))  # Preserve raw FAISS distance
            for dist, idx in zip(distances[0], indices[0])
            if dist <= distance_threshold and idx != -1
        ]

        print(f"Search complete. Index: '{self.index_name}' | Query: '{query}' | Results: {faiss_results}")
        return faiss_results  # Returns [(doc_id, distance)]
