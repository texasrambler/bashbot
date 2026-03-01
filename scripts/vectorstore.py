# %% [markdown]
# ### VectorSrore Database

# %%
import os
import sys
import numpy as np
import uuid
import shutil
from pathlib import Path
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
from chromadb.utils.batch_utils import create_batches
from typing import List, Dict, Any, Tuple
from sklearn.metrics.pairwise import cosine_similarity
from langchain_core.documents import Document
from langchain_community.document_loaders import DirectoryLoader, TextLoader #,PyPDFLoader, PyMuPDFLoader for PDFs in the future
from langchain_text_splitters import RecursiveCharacterTextSplitter

# %%
class VectorStore:
    """Manages embeddings in a vector database"""

    def __init__(self, collection_name: str = "bashdocs", persist_directory: str = "../data/vector_store"):
        """
        ctor

        Args:
            collection_name: Name of the collection
            persist_directory: Path to directory for persistant vectors.
        """
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        self.client = None
        self.collection = None
        self._initialize_store()

    def remove_store(self):
        """Removes any existing local db store files"""
        # First remove any existing store if one is already there
        store_path = Path(self.persist_directory)
        if store_path.exists() and store_path.is_dir():
            shutil.rmtree(store_path)
            
        
    def _initialize_store(self):
        """Initialize the ChromaDB client and collection"""
        try:
            # Create the persistent ChromaDB client
            os.makedirs(self.persist_directory, exist_ok=True)
            self.client = chromadb.PersistentClient(path=self.persist_directory)

            # Get or create the collection
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"description": "Bash commands document embeddings for RAG"}
            )
            print(f"Vector store insitialized for collection {self.collection_name}")
            print(f"Exisiting documents in collection: {self.collection.count()}")

        except Exception as e:
            print(f"Error initializing vector store: {e}")
            raise
            
    def add_documents(self, documents: List[Any], embeddings: np.ndarray):
        """
        Add documents to the vector store

        Args:
            documents: List of langchain Documents
            embeddings: The embeddings for the documents
        """
        if len(documents) != len(embeddings):
            raise ValueError("Number of documents to embeddings mismatch.")

        print(f"Adding {len(documents)} documents to the vector store...")

        # Prepare data for ChromaDB
        ids = []
        metadatas = []
        documents_text = []
        embeddings_list = []

        for idx, (doc, embedding) in enumerate(zip(documents, embeddings)):
            # Generate the uuid
            doc_id = f"doc_{uuid.uuid4().hex[:8]}_{idx}"
            ids.append(doc_id)

            # Metadata
            metadata = dict(doc.metadata)
            metadata['doc_index'] = idx
            metadata['content_length'] = len(doc.page_content)
            metadatas.append(metadata)

            # Document Page Content
            documents_text.append(doc.page_content)

            # Embedding vetor
            embeddings_list.append(embedding.tolist())

        # Add it to the collection
        try:
            max_batch = 5000 # batch max is 5461
            for i in range(0, len(embeddings), max_batch):
                self.collection.add(
                    embeddings=embeddings_list[i:i + max_batch],
                    documents=documents_text[i:i + max_batch],
                    ids=ids[i:i + max_batch],
                    metadatas=metadatas[i:i + max_batch]
                )
            print(f"Successfully added {len(documents)} documents to the store.")
            print(f"Total documents in collection: {self.collection.count()}")

        except Exception as e:
            print(f"Error adding documents to vector store: {e})")
            raise
            
