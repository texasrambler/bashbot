import numpy as np
import chromadb
import ollama
from typing import List, Dict, Any, Tuple
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_classic.chains import RetrievalQA
from langchain_community.llms import Ollama

class EmbeddingMgr:
    """This class manages the embedding implementation.
    """

    def __init__(self, embedding_model_name: str = "nomic-embed-text") -> List[Any]:
        """
        ctor

        Args:
            model_name: embedding model name for embeddings.

        Returns:
            List of embeddings
        """

        self.embedding_model_name = embedding_model_name
        self.model = None
        self._load_components()
        
    def _load_components(self):
        """This method loads the embedding model defined in the ctor."""
        try:
            print(f"Loading Embedding model {self.embedding_model_name}.")
            self.model = OllamaEmbeddings(model=self.embedding_model_name)
            print(f"Embedding Model {self.model} loaded successfully.")
        except Exception as e:
            print(f"Exception while loading {self.embedding_model_name}: {e}")
            raise
        
    def add_embeddings(self, texts):
        """
        Generate embeddings for a list of text strings

        Args:
            texts: List of strings
        """
        if not self.model:
            raise ValueError("Model has not been loaded.")
        
        print(f"Generating embeddings for {len(texts)} strings.")

        embeddings = []
        for chunk in texts:
        # generate embeddings
            embedding = ollama.embeddings(
                model= self.embedding_model_name,
                prompt=chunk.page_content
            )["embedding"]

            embeddings.append(embedding)

        return embeddings
