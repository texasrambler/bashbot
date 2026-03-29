#!/usr/bin/env python
# %% [markdown]
# # Source Code for bashbot.py
#
# This notebook is used by jupytext to convert to and from a Python script _'jupytext --from notebook --to py --output ../scripts/bashbot.py bashbot.ipynb'_

# %%
import os
import sys
import argparse
from pathlib import Path
from langchain_core.documents import Document
from langchain_community.document_loaders import DirectoryLoader, TextLoader #,PyPDFLoader, PyMuPDFLoader for PDFs in the future
from langchain_text_splitters import RecursiveCharacterTextSplitter
from vectorstore import VectorStore
from embeddings import EmbeddingMgr
from retriever import Retriever
from langchain_groq import ChatGroq

# %% [markdown]
# ## Function Definitions

# %% [markdown]
# ### load the corpus from the text files in the data directory

# %% [markdown]
# ### process text _(*.txt)_ files in the data directory

# %%
def process_text_files(path):
    """Processes all text (.txt) files in the provided path"""
    documents = []
    file_dir = Path(path)

    files = list(file_dir.glob("**/*.txt"))
    print(f"Preparing to process {len(files)} text files...")

    # process the files and modify the source metadata to include manpage reference if possible.
    for f in files:
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

# %% [markdown]
# ### wrapper to provide the path to the data direcory

# %%
def process_all_files():
    """wrapper to provide the path to the data direcory"""
    documents = []
    documents.extend(process_text_files("../data"))

    return documents

# %% [markdown]
# ### divide text into chunks

# %%
def split_documents(corpus, chunk_size=500, chunk_overlap=200):
    """splits the douments into manageable sized chunks"""
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
# ### Create embeddings

# %%
def create_embeddings(vector_store: VectorStore, emb_mgr: EmbeddingMgr):
    """
    wrapper to create embeddings and store in the vector store.
    It will create duplicates using the uuid, so to prevent this,
    call remove_store() on the VectorStore instance
    """
    corpus = process_all_files()
    chunks = split_documents(corpus)

    # get the text from the documents
    texts = [doc.page_content for doc in chunks]
    
    # Use the EmbeddingMgr class to generate them
    embeddings = emb_mgr.generate_embeddings(texts)
    
    # Add the documents to the vector store
    vector_store.add_documents(chunks, embeddings)

# %% [markdown]
# ### makes the RAG call to the llm _(currently Groq)_

# %%
# RAG function to pull it all together
def rag_call(query, retriever, llm, top_k=3):
    fetched_docs = retriever.fetch(query, top_k=top_k)

    if not fetched_docs:
        return "No relevant documents found."

    # put all fetched documents into the context
    context = "\n\n".join(doc['content'] for doc in fetched_docs)

    # Create the prompt
    prompt = f"""You are a proffesional AI assistant theat specializes in asnwering
        questions about Linux commands.
        Use the following context to answer the question accurately and concisely.
        Context:
        {context}
        
        Question: {query}
        
        Answer:"""

    response = llm.invoke([prompt.format(context=context, query=query)])
    return response.content

# %% [markdown]
# ### prompt for a single question and return the answer

# %%
def prompt_for_question(fido: Retriever, llm) -> str:
    """prompt for a single question and return the answer"""
    question = input()
    if question.upper() == "QUIT":
        return "QUIT"
    answer = rag_call(question, fido, llm)
    return answer

# %% [markdown]
# ### main()

# %%
def main():
    """
    Main program to run bashbot

    Args:
        run_mode: [optional]
    """
    
    # Check for 3.13 minimum python version
    if sys.version_info.major != 3 or sys.version_info.minor < 10:
        print("python version 3.10 or greater is required.")
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
    emb_mgr = EmbeddingMgr()
    vector_store = VectorStore()

    # Check run mode
    if args.delete:
        print("Deleting the vector store...")
        vector_store.remove_store()                 # deletes the current store
    elif args.add:
        print("Adding new data...")
        create_embeddings(vector_store, emb_mgr)    # adds documents to store
    else:
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            print("GROQ_API_KEY is not set. Please export your Groq key in the environment.")
            return

        llm = ChatGroq(groq_api_key=api_key, model_name="llama-3.1-8b-instant", temperature=0.1, max_tokens=1024)
        fido = Retriever(vector_store, emb_mgr)

        while True:
            # Prompt for question
            print("\nEnter your question ('quit' to exit):")
            try:
                result = prompt_for_question(fido, llm)
            except Exception as e:
                print(f"An error occurred {e}")
                return

            # Look for quit
            if result == "QUIT":
                print("Goodbye.")
                return

            print(result)

# %%
if __name__ == "__main__":
   main()

