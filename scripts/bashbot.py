#!/usr/bin/env python
# %% [markdown]
# # Source Code for bashbot.py
#
# This notebook is used by jupytext to convert to and from a Python script
#
# `jupytext --from notebook --to py --output ../scripts/bashbot.py bashbot.ipynb`

# %%
import sys
import argparse
import shutil

import ollama
from typing import Tuple
from pathlib import Path
from vectorstore import VectorStore
from embeddings import EmbeddingMgr
from loader import Loader

# %% [markdown]
# ## Function Definitions

# %% [markdown]
# ### prompt for a single question and return the answer

# %%
def get_relevant_docs(query, embedding_manager, vector_store, top_k=10, verbose=False, threshold=0.6):
    # Get embedding for question
    question_embedding = ollama.embeddings(
        model=embedding_manager.embedding_model_name, prompt=query
    )["embedding"]

    # Find relevant chunks
    query_results = vector_store.collection.query(
        query_embeddings=[question_embedding], n_results=top_k
    )


    if verbose:
#        print(query_results)
        print(f"filtering based on threshold {threshold}...")

    filtered_results = []
    for i, distance in enumerate(query_results['distances'][0]):
        similarity = 1 - distance # Convert to similarity
        if similarity >= threshold:
            filtered_results.append({
                "id": query_results['ids'][0][i],
                "document": query_results['documents'][0][i],
                "metadata": query_results['metadatas'][0][i],
                "similarity": similarity
            })

    if verbose:
        print(filtered_results)

    return filtered_results


# %%
def send_question(
    query, llm, embedding_manager, vector_store, top_k=10, pure: bool = False, verbose=False, threshold=0.6
) -> Tuple:
    context = ""
    found = 0

    if not pure:
        results = get_relevant_docs(query, embedding_manager, vector_store, top_k, verbose, threshold)
        found = len(results)
        if found > 0:
            # Build context from retrieved chunks
            documents = []
            for res in results:
                documents.append(res["document"])
            context = "\n\n".join(documents)
        else:
            return (0, "No relevant documents found.")

        prompt = f"""You are a professional AI assistant that specializes in answering
            questions about Linux commands.
            Use ONLY the following context to answer the question accurately and concisely.

            Context:
            {context}

            Question: {query}

            Answer:"""
    else:
        prompt = f"""You are a professional AI assistant that specializes in answering
            questions about Linux commands.

            Context:
            {context}

            Question: {query}

            Answer:"""
    # generate answer
    response = ollama.chat(model=llm, messages=[{"role": "user", "content": prompt}])

    return (found, response["message"]["content"])


# %%
def prompt_for_question(llm_name, embeddingmgr, vector_store, verbose=False, threshold=0.6) -> Tuple:
    """prompt for a single question and return the answer"""
    question = input()
    if question.upper() == "QUIT":
        return (0, "QUIT")

    return send_question(question, llm_name, embeddingmgr, vector_store, verbose=verbose, threshold=threshold)

# %%
def format_results(results: Tuple) -> str:
    header = "==== Documents Found ===="
    separator = "======== Answer ========="
    footer = "-------------------------"
    docs, answer = results
    found = f"Found {docs} documents."
    return f"\n{header}\n\n{found}\n\n{separator}\n\n{answer}\n\n{footer}\n"


# %% [markdown]
# ### main()

# %%
def main(llm_name: str = "granite3.3:8b"):
    """
    Main program to run bashbot

    Args:
        llm_name: [optional]
    """

    # Check for 3.13 minimum python version
    if sys.version_info.major != 3 or sys.version_info.minor < 12:
        print("python version 3.12 or greater is required.")
        return None

    # Parse commandline for args
    parser = argparse.ArgumentParser(
        prog="bashbot",
        description="RAG chatbot agent for linux commands.",
        epilog="Duplicate records are possible if run with --add flag.",
    )

    parser.add_argument(
        "-d",
        "--delete",
        action="store_const",
        const="delete",
        help="deletes the vector store",
    )  # delete the current db
    parser.add_argument(
        "-a",
        "--add",
        action="store_const",
        const="add",
        help="add entries to the vector store",
    )  # add records to the db
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_const",
        const="verbose",
        help="emit verbose output",
    )  # verbose output
    parser.add_argument(
        "-t",
        "--threshold",
        type=float,
        default=0.6, help="The similarity threshold (defaults to 0.6)"
    )  # threshold

    args = parser.parse_args()

    # Parser passed so initialize the store and manager
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

        # Create a Loader
        loader = Loader()

        # Process the files
        print("Processing files.")
        loader.process_files()

        # Combine chunks into a corpus for embeddings
        texts = [doc for doc in loader.chunks]

        # Use the EmbeddingMgr class to generate them
        print("Generating embeddings...")
        embeddings = emb_mgr.add_embeddings(texts)

        # Add the documents to the vector store
        print("Inserting into vector store...")
        vector_store.add_documents(loader.chunks, embeddings)
        return 0
    else:
        print("Preparing LLM...")

        while True:
            # Prompt for question
            print("\nEnter your question ('quit' to exit):")
            try:
                results = prompt_for_question(
                    llm_name=llm_name, embeddingmgr=emb_mgr, vector_store=vector_store, verbose=args.verbose, threshold=args.threshold
                )
                _, answer = results
            except Exception as e:
                print(f"An error occurred {e}")
                return 1

            # Look for quit
            if answer == "QUIT":
                print("Goodbye.")
                return 0

            if args.verbose:
                print(format_results(results))
            else:
                _, answer = results
                print(f"\n{answer}\n")

# %%
if __name__ == "__main__":
    main()


# %%
