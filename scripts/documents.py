#!/usr/bin/env python
# %% [markdown]
# # Documents Notebook
#
# For this chatbot I will be using bash man pages, tealdeer output and the output from the bash help system for builtin commands.
#
# There will need to be 2 pipelines.
# - The first will be the data ingestion pipeline, which is in this notebook.
# - Then we will need a query pipeline that will use our domain specific emnbeddings as the context for an LLM prompt. This is the RAG implementation that will be built. It is located in bashbot.ipynb and the associated script.
#
# The dependencies are in requirements.txt.

# %% [markdown]
# ### Architecture

# %% [markdown]
# ![image](../images/architecture.png)

# %% [markdown]
# ## Data Ingestion Pipeline for Bashbot

# %% [markdown]
# ### Imports

# %%
import os
import sys
from pathlib import Path
from langchain_core.documents import Document
from langchain_community.document_loaders import DirectoryLoader, TextLoader #,PyPDFLoader, PyMuPDFLoader for PDFs in the future
from langchain_text_splitters import RecursiveCharacterTextSplitter


# %% [markdown]
# ## Function Definitions

# %% [markdown]
# ### load the corpus from the text files in the data directory

# %%
def load_txt_corpus():
    """ Loads all text files from the ./data directory."""
    loader = DirectoryLoader(
        "../data",
        glob="**/*.txt",
        loader_cls=TextLoader,
        loader_kwargs={'encoding': 'utf-8'},
        show_progress=True
    )
    corpus = loader.load()

    return corpus

# %%
def process_text_files(path):
    """Processes all text (.txt) files in the provided path"""
    documents = []
    file_dir = Path(path)

    files = list(file_dir.glob("**/*.txt"))
    print(f"Preparing to process {len(files)} text files...")

    # process the files and modify the source metadata to include manpage reference if possible.
    for f in files:
#        print(f"processing {f.name}")
        try:
            loader = TextLoader(str(f))
            docs = loader.load()

            for d in docs:
                d.metadata['manpage'] = Path(f).stem
                d.metadata['source_file'] = f.name
                d.metadata['file_type'] = 'txt'
                d.metadata['platform'] = 'linux'
                d.metadata['source'] = 'manpage'

                documents.extend(docs)

        except Exception as e:
            print(f"    ERROR {e}")

    print(f"Total documents loaded: {len(documents)}.")
    print(type(documents))
    return list(documents)
    


# %%
def process_all_files():
    documents = []
    documents.extend(process_text_files("../data"))

    return documents


# %% [markdown]
# ### divide text into chunks

# %%
# Splits the douments into manageable sized chunks.
def split_documents(corpus, chunk_size=500, chunk_overlap=200):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", " ", ""]
    )
    split_docs = text_splitter.split_documents(corpus)
    print(f"Splitting {len(corpus)} documents into {len(split_docs)} chunks.")
    if split_docs:
        print("\nSample chunk:")
        print(f"Page_content: {split_docs[0].page_content[:300]}...")
        print(f"Metadata: {split_docs[0].metadata}")

    return split_docs

# %% [markdown]
# ### main()

# %%
# Obligatory entry point
def main():
   # load_txt_corpus()
    corpus = process_all_files()

# %%
#if __name__ == "__main__":
#   main()

# %% [markdown]
# ## Notebook Exploration

# %% [markdown]
# **Read data from files, and maybe an RDB**

# %%
corpus = process_all_files()

# %% [markdown]
# **Chunking**

# %%
chunks = split_documents(corpus)

# %%
chunks[3]

# %% [markdown]
# ### Embedding Manager

# %% [markdown]
# I want the flexibility to have pluggable vector store implementations. I am going to wrap this part in classes.
# Inspiration for this came from Krish Naik's YouTube video
# https://youtu.be/o126p1QN_RI?si=7xC5H47A3iuu52RK
# and pixegami https://youtu.be/2TJxpyO3ei4?si=zPNEsAFWWy5Cenzq
#

# %%
import numpy as np
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
from chromadb.utils.batch_utils import create_batches
import uuid
from typing import List, Dict, Any, Tuple
from sklearn.metrics.pairwise import cosine_similarity


# %%
class EmbeddingMgr:
    """This class manages the embedding implementation.
    For now I am going to use a SentenceTransformer of
    some kind."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        ctor

        Args:
            model_name: model name for sentence embeddings.

        Notes: For now I am using HuggingFace models.
        """

        self.model_name = model_name
        self.model = None
        self._load_model()
        
    def _load_model(self):
        """This method loads the model defined in the ctor."""
        try:
            print(f"Loading model {self.model_name}.")
            self.model = SentenceTransformer(self.model_name)
            print(f"Model loaded successfully.")
            print(f"Embedding dimension: {self.model.get_sentence_embedding_dimension()}")
        except Exception as e:
            print(f"Exception while loading model {self.model_name}: {e}")
            raise
        
    def generate_embeddings(self, texts: List[str]) -> np.ndarray:
        """
        Generate embeddings for a list of text strings

        Args:
            texts: List of strings

        Returns:
            numpy array of embeddings with shape (len(texts), embedding_dim)
        """
        if not self.model:
            raise ValueError("Model has not been loaded.")

        print(f"Generating embeddings for {len(texts)} strings.")
        embeddings = self.model.encode(texts, show_progress_bar=True)
        print(f"Generated embeddings with shape: {embeddings.shape}")
        return embeddings
    


# %% [markdown]
# Initialize the Embedding Manager

# %%
emb_mgr = EmbeddingMgr()
emb_mgr


# %% [markdown]
# ### VectorSrore Database

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
            


# %%
vector_store = VectorStore()
vector_store

# %% [markdown]
# ### Create embeddings

# %%
# get the text from the documents
texts = [doc.page_content for doc in chunks]

# Use the EmbeddingMgr class to generate them
embeddings = emb_mgr.generate_embeddings(texts)

# Add the documents to the vector store
vector_store.add_documents(chunks, embeddings)


# %% [markdown]
# ## Prompt and RAG Retrieval Pipeline

# %%
