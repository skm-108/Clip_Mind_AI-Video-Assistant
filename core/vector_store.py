import os 

CHROMA_DIR = "vector_db"
COLLECTION_NAME = "meeting_transcript"
EMBEDDING_MODEL  = "all-MiniLM-L6-v2"

def get_embeddings():
    try:
        from langchain_huggingface import HuggingFaceEmbeddings
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "langchain-huggingface is required for embeddings and vector storage. Install dependencies with 'pip install -r requirements.txt'."
        ) from exc

    return HuggingFaceEmbeddings(
        model_name = EMBEDDING_MODEL,
        model_kwargs = {"device" : 'cpu'}
    )

def build_vector_store(transcript : str):
    from langchain_chroma import Chroma
    try:
        from langchain_text_splitters import RecursiveCharacterTextSplitter
    except ModuleNotFoundError:
        RecursiveCharacterTextSplitter = None
    from langchain_core.documents import Document

    print("Building vector Store")

    if RecursiveCharacterTextSplitter is not None:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
        )
        chunks = splitter.split_text(transcript)
    else:
        # Fallback: naive sentence/paragraph splitter for embeddings
        pieces = [p.strip() for p in __import__('re').split(r"\n{2,}", transcript) if p.strip()]
        if not pieces:
            pieces = [s.strip() for s in __import__('re').split(r'(?<=[.!?]) +', transcript) if s.strip()]
        chunks = []
        current = ""
        for piece in pieces:
            if len(current) + len(piece) + 1 <= 500:
                current = (current + " " + piece).strip()
            else:
                if current:
                    chunks.append(current)
                if len(piece) <= 500:
                    current = piece
                else:
                    for i in range(0, len(piece), 500 - 50):
                        chunks.append(piece[i : i + (500 - 50)])
                    current = ""
        if current:
            chunks.append(current)

    docs = [
        Document(page_content=chunk, metadata = {'chunk_index' : i})
        for i,chunk in enumerate(chunks)
    ]

    embeddings = get_embeddings()
    vector_store = Chroma.from_documents(
        documents= docs,
        embedding=embeddings,
        collection_name=COLLECTION_NAME,
        persist_directory=CHROMA_DIR
    )

    return vector_store



def load_vector_store():
    from langchain_chroma import Chroma

    embeddings = get_embeddings()
    vector_store = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function= embeddings,
        persist_directory=CHROMA_DIR
    )

    return vector_store

def get_retriever(vector_store, k :int = 4):
    return vector_store.as_retriever(
        search_type = 'similarity',
        search_kwargs = {"k":k}
    )


