#!/usr/bin/env python
# %% [markdown]
# # Bashbot Notebook

# %% [markdown]
# ## Imports

# %%
import os
import sys
import shutil
import chromadb
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
from langchain.schema import Document
from langchain_openai import OpenAIEmbeddings
#from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from openai import OpenAI

# %% [markdown]
# ## Globals

# %%
# Globals need to be retrieved from cmdline if time permits
# A DEBUG global boolean is currently used to print debug info.
# This needs to be replaced with a proper logging framework.
DB_CHROMA = True
DEBUG = False
DOC_PATH = "data"
DB_PATH = "db/chroma"
GLOB = "*.txt"

# %% [markdown]
# ## Function Definitions

# %% [markdown]
# ### load the corpus

# %%
# Loads the documents fro the path defined in DOC_PATH
# filename globbing is performed using the pattern defined in GLOB
def load_corpus():
    loader = DirectoryLoader(DOC_PATH, GLOB)
    corpus = loader.load()
    print(type(corpus))
    print(type(corpus[1]))
    return corpus

# %% [markdown]
# ### divide text into chunks

# %%
# Splits the douments into manageable sized chunks.
def split_docs(corpus):
    splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=200,
            length_function=len,
            add_start_index=True,
            )
    splits = splitter.split_documents(corpus)
    print(f"Splitting {len(corpus)} documents into {len(splits)} chunks.")

    return splits

# %% [markdown]
# ### save to a Chroma vector database

# %%
# Currently using the Chroma vector database and saving locally.
def chroma_save(corpus):
    client = chromadb.PersistentClient(path=DB_PATH)
#    collection = client.create_collection(
#        name="bashbot",
#        embedding_function=OpenAIEmbeddingFunction(
#            api_key=os.getenv("OPENAI_API_KEY"),
#            model_name="text-embedding-3-small"
#       )
#    )
#    client.add(corpus)

    if DEBUG:
        print(f"Saved {len(corpus)} documents to {DB_PATH}.")

# %% [markdown]
# ### wrapper function to allow subsitution of the vector db implementation

# %%
# Wrapper function to allow the vector database to be replaced
# with new options.
def save_to_db(chunks):
    if os.path.exists(DB_PATH):
        shutil.rmtree(DB_PATH)

    if DB_CHROMA:
        chroma_save(chunks)
    else:
        print("Database type not found.", file=sys.stderr)

# %% [markdown]
# ### generate the data and store in db

# %%
# Processes the documents in the DOC_PATH directory and saves
# them to the vector database.
def gen_data_db():
    corpus = load_corpus()

    if DEBUG:
        print(len(corpus))
        
    # chunks = chunk_docs(corpus)
    # save_to_db(chunks)

# %% [markdown]
# ### main()

# %%
# Obligatory entry point
def main():
   gen_data_db()

# %%
if __name__ == "__main__":
   main()

# %% [markdown]
# ## Notebook Exploration

# %%
corpus = load_corpus()

# %%
print(corpus[0])

# %%
print(len(corpus))

# %%
docs = split_docs(corpus)

# %%
len(docs)

# %%
docs[42]

# %%
corpus[0] 

# %%
# save_to_db(corpus)

# %%
client = chromadb.Client()

#collection = client.get_or_create_collection(name="bashbot")


# %%
import chromadb
# setup Chroma in-memory, for easy prototyping. Can add persistence easily!
client = chromadb.Client()

# Create collection. get_collection, get_or_create_collection, delete_collection also available!
collection = client.create_collection("all-my-documents")

# Add docs to the collection. Can also update and delete. Row-based API coming soon!
collection.add(
    documents=["This is document1", "This is document2"], # we handle tokenization, embedding, and indexing automatically. You can skip that and add your own embeddings as well
    metadatas=[{"source": "notion"}, {"source": "google-docs"}], # filter on these!
    ids=["doc1", "doc2"], # unique for each doc
)

# Query/search 2 most similar results. You can also .get by id
results = collection.query(
    query_texts=["This is a query document"],
    n_results=2,
    # where={"metadata_field": "is_equal_to_this"}, # optional filter
    # where_document={"$contains":"search_string"}  # optional filter
)

# %%
