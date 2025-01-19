import argparse
import os
import shutil
from pathlib import Path
from langchain_community.document_loaders import PyPDFDirectoryLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema.document import Document
from models import model
from langchain_chroma import Chroma

# Defines the flags of the program.
pd = argparse.ArgumentParser()
pd.add_argument("--reset", action="store_true", help="Reset the database.") # Reset the database, which is required to erase data or use a different model.

pd.add_argument("--llm_model", type=str, help="Select the model of the llm. (Default: llama3:latest)", default="llama3:latest")

# One of them is mandatory but can't execute both at the same time.
doctype = pd.add_mutually_exclusive_group()
doctype.add_argument("--pdf", action="store_true", help="Set the doctype to pdf.")
doctype.add_argument("--md", action="store_true", help="Set the doctype to md.")
args_pd = pd.parse_args()

pd_model = model(llm=args_pd.llm_model)

CHROMA_PATH = "chroma"
DATA_PATH = ""
if args_pd.pdf:
    DATA_PATH = "data/pdf"
if args_pd.md:
    DATA_PATH = "data/md"

def main():

    if args_pd.reset:
        print("✨ Clearing Database.")
        clear_database()
        print("🧘 Database Cleared.")
        exit()

    # Create (or update) the data store.
    documents = load_documents()
    chunks = split_documents(documents)
    add_to_chroma(chunks)


def load_documents():
    if args_pd.pdf:
        return PyPDFDirectoryLoader(DATA_PATH).load()
    if args_pd.md:
        return DirectoryLoader(DATA_PATH, glob="*.md").load()

def split_documents(documents: list[Document]):
    return RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=80, length_function=len, is_separator_regex=False).split_documents(documents)

def add_to_chroma(chunks: list[Document]):
    # Load the existing database.
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=pd_model.embeddings)
    # Calculate Page IDs.
    chunks_with_ids = calculate_chunk_ids(chunks)
    # Add or Update the documents.
    existing_items = db.get(include=[])  # IDs are always included by default
    existing_ids = set(existing_items["ids"])
    print(f"Number of existing documents in DB: {len(existing_ids)}")
    # Only add documents that don't exist in the DB.
    new_chunks = []
    for chunk in chunks_with_ids:
        if chunk.metadata["id"] not in existing_ids:
            new_chunks.append(chunk)

    if len(new_chunks):
        print(f"👉 Adding new documents: {len(new_chunks)}")
        new_chunk_ids = [chunk.metadata["id"] for chunk in new_chunks]
        db.add_documents(new_chunks, ids=new_chunk_ids)
    else:
        print("✅ No new documents to add")

def calculate_chunk_ids(chunks):
    # This will create IDs like "data/monopoly.pdf:6:2"
    # Page Source : Page Number : Chunk Index
    last_page_id = None
    current_chunk_index = 0

    for chunk in chunks:
        source = chunk.metadata.get("source")
        page = chunk.metadata.get("page")
        current_page_id = f"{source}:{page}"
        # If the page ID is the same as the last one, increment the index.
        if current_page_id == last_page_id:
            current_chunk_index += 1
        else:
            current_chunk_index = 0
        # Calculate the chunk ID.
        chunk_id = f"{current_page_id}:{current_chunk_index}"
        last_page_id = current_page_id
        # Add it to the page meta-data.
        chunk.metadata["id"] = chunk_id
    return chunks

def clear_database():
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)

if __name__ == "__main__":
    main()