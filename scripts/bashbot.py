#!/usr/bin/env python
# %% [markdown]
# # Source Code for bashbot.py
#
# This notebook is used by jupytext to convert to and from a Python script _'jupytext --from notebook --to py --output ../scripts/bashbot.py bashbot.ipynb'_

# %%
import os
import sys
import argparse
import shutil
import ollama
from typing import List, Dict, Any, Tuple
from pathlib import Path
from langchain_core.documents import Document
from langchain_community.document_loaders import DirectoryLoader, TextLoader #,PyPDFLoader, PyMuPDFLoader for PDFs in the future
from langchain_text_splitters import RecursiveCharacterTextSplitter
from vectorstore import VectorStore
from embeddings import EmbeddingMgr
from loader import Loader
from ollama import chat

# %% [markdown]
# ## Function Definitions

# %% [markdown]
# ### prompt for a single question and return the answer

# %%
def get_relavent_docs(query, embedding_manager, vector_store, top_k=3):
    # Get embedding for question
    question_embedding = ollama.embeddings(
        model=embedding_manager.embedding_model_name,
        prompt=query
    )["embedding"]

    # Find relevant chunks
    results = vector_store.collection.query(
        query_embeddings=[question_embedding],
        n_results=top_k
    )         
#    print(results)
    return results


# %%
def send_question(query, llm, embedding_manager, vector_store, top_k=3, pure: bool = False) -> Tuple:
    contex = ""
    found = 0

    if not pure:
        results = get_relavent_docs(query, embedding_manager, vector_store, top_k)
        found = results["ids"]
        if len(found[0]) > 0:    
            # Build context from retrieved chunks
            context = "\n\n".join(results["documents"][0])
        else:
            return ("No relavent documents found.", 0)   

    prompt = f"""You are a professional AI assistant that specializes in answering
        questions about Linux commands.
        Use ONLY the following context to answer the question accurately and concisely.
        
        Context:
        {context}
        
        Question: {query}
        
        Answer:"""
 
    # generate answer 
    response = ollama.chat(
        model=llm,
        messages=[{
            "role": "user",
            "content": prompt
        }]
    )

    return (found, response["message"]["content"])


# %%
def prompt_for_question(llm_name, embeddingmgr, vector_store) -> Tuple:
    """prompt for a single question and return the answer"""
    question = input()
    if question.upper() == "QUIT":
        return "QUIT"

    return send_question(question, llm_name, embeddingmgr, vector_store)

# %% [markdown]
# ### main()

# %%
def main(llm_name: str = "granite3.3:8b"):
    """
    Main program to run bashbot

    Args:
        run_mode: [optional]
    """

    # Check for 3.13 minimum python version
    if sys.version_info.major != 3 or sys.version_info.minor < 12:
        print("python version 3.12 or greater is required.")
        return
    
    # Parse commandline for args
    parser = argparse.ArgumentParser(
                    prog='bashbot',
                    description='RAG chatbot agent for linux commands.',
                    epilog='Duplicate records are possible if run with --add flag.')

    parser.add_argument('-d', '--delete', action='store_const', const='delete', help='deletes the vector store')    # delete the current db 
    parser.add_argument('-a', '--add', action='store_const', const='add', help='add entries to the vector store')   # add records to the db

    args = parser.parse_args()

    # Parser passed so initialize the store and manager
    loader = None
    emb_mgr = EmbeddingMgr()
    vector_store = VectorStore()

    # Check run mode
    if args.delete:
        print("Deleting the vector store...")
        store_path = Path("../data/vector_store/")
        print(f"Removing vector store at {store_path}.")
        shutil.rmtree(store_path, ignore_errors=True) 
        print("Store deleted. Rerun the application with -a to create with new data.")
        return 0   
    elif args.add:
        print("Adding new data...")
        loader = Loader()
        loader.process_files()
    else:
        print("Preparing LLM...")

        while True:
            # Prompt for question
            print("\nEnter your question ('quit' to exit):")
            try:
                docs_found, result = prompt_for_question(llm_name=llm_name, embeddingmgr=emb_mgr, vector_store=vector_store)
            except Exception as e:
                print(f"An error occurred {e}")
                return

            # Look for quit
            if result == "QUIT":
                print("Goodbye.")
                return

            print(f"found {len(docs_found[0])}\n{docs_found}\n{result}")

# %%
if __name__ == "__main__":
   main()


# %%
