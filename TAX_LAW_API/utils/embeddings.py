from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from utils.config import OPENAI_API_KEY, PINECONE_API_KEY, EMBEDDING_MODEL

def create_embedding_model():
    return OpenAIEmbeddings(
        model=EMBEDDING_MODEL,
        openai_api_key=OPENAI_API_KEY,
    )

def create_vector_store(index_name: str):
    embedding_model = create_embedding_model()
    return PineconeVectorStore(
        index_name=index_name,
        embedding=embedding_model,
        pinecone_api_key=PINECONE_API_KEY
    ) 