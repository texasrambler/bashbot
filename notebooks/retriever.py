import numpy as np
from vectorstore import VectorStore
from embeddings import EmbeddingMgr
from typing import List, Dict, Any, Tuple
from sklearn.metrics.pairwise import cosine_similarity
from langchain_core.documents import Document
from sentence_transformers import SentenceTransformer

class Retriever:
    """Encodes a query and returns similar documents from the vector store db"""
    
    def __init__(self, vector_store: VectorStore, embedding_manager: EmbeddingMgr):
        """
        ctor

        Args:
            vector_store: The VectorStore of embeddings
            embedding_manager: The EmbeddingMgr that handles encoding embeddings
        """
        self.store = vector_store
        self.manager = embedding_manager

    def fetch(self, query: str, top_k: int = 5, threshold: float = 0.0) -> List[Dict[str, Any]]:
        """
        Retrieves documents from the store

        Args:
            query: The user query
            top_k: Number of results to return
            threshold: Minimum similarity score

        Returns:
            List of dictionaries with the documents and metadata
        """
        print(f"Fetching documents for '{query}'")
        print(f"Top K: {top_k}. Threshold: {threshold}")

        # create the embedding for the query
        query_embedding = self.manager.generate_embeddings([query])[0]

        # Search the vector store
        try:
            results = self.store.collection.query(
                query_embeddings=[query_embedding.tolist()],
                n_results=top_k
            )

            fetched_docs = []

            if results['documents'] and results['documents'][0]:
                documents = results['documents'][0]
                metadatas = results['metadatas'][0]
                distances = results['distances'][0]
                ids = results['ids'][0]

                for i, (doc_id, document, metadata, distance) in enumerate(zip(ids, documents, metadatas, distances)):
                    # Convert distance to similarity score - cosine distance for chromadb
                    similarity = 1 - distance

                    if similarity >= threshold:
                        fetched_docs.append({
                            'id': doc_id,
                            'content': document,
                            'metadata': metadata,
                            'similarity_score': similarity,
                            'distance': distance,
                            'rank': i + 1
                        })

                print(f"Fetched {len(fetched_docs)} documents (filtered by threshold)")
            else:
                print("No documents found")

            return fetched_docs

        except Exception as e:
            print(f"Error fetching documents: {e}")
            return []
            


