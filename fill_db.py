# fill_db.py
import os
import shutil
import chromadb
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import config  # Importing your central config

def fill():
    """
    Clears the existing database and refills it with fresh data 
    from the PDF directory defined in config.py.
    """
    print(f"🔄 Preparing to ingest documents from '{config.DATA_PATH}'...")

    # 1. Initialize Client
    chroma_client = chromadb.PersistentClient(path=config.CHROMA_PATH)

    # 2. Reset Collection (Prevent Duplicates)
    # We try to delete the collection first so we start fresh.
    try:
        chroma_client.delete_collection(name=config.COLLECTION_NAME)
        print("   - Existing collection cleared.")
    except ValueError:
        pass  # Collection didn't exist yet, which is fine.

    collection = chroma_client.get_or_create_collection(name=config.COLLECTION_NAME)

    # 3. Load Documents
    if not os.path.exists(config.DATA_PATH):
        os.makedirs(config.DATA_PATH)
        print(f"   - Warning: '{config.DATA_PATH}' was missing. Created it. Please put PDFs there.")
        return

    loader = PyPDFDirectoryLoader(config.DATA_PATH)
    raw_documents = loader.load()

    if not raw_documents:
        print("   - ⚠️ No PDF documents found. Database is empty.")
        return

    # 4. Split Documents
    print(f"   - Splitting {len(raw_documents)} pages...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=100,
        length_function=len,
        is_separator_regex=False,
    )
    chunks = text_splitter.split_documents(raw_documents)

    # 5. Prepare Data for DB
    documents = []
    metadatas = []
    ids = []

    for i, chunk in enumerate(chunks):
        documents.append(chunk.page_content)
        metadatas.append(chunk.metadata)
        ids.append(f"ID_{i}")

    # 6. Insert into ChromaDB
    collection.upsert(
        documents=documents,
        metadatas=metadatas,
        ids=ids
    )

    print(f"✅ Success! Added {len(documents)} chunks to the database.")

if __name__ == "__main__":
    fill()