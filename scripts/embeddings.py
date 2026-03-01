
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
import uuid
import os
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
from chromadb.utils.batch_utils import create_batches
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
    
