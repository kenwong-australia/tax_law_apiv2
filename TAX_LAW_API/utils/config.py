import os
from dotenv import load_dotenv, find_dotenv

# Load environment variables
env_path = find_dotenv()
if not env_path:
    print("WARNING: .env file not found")
load_dotenv(env_path)

# API Keys and Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "taxlawlegato")

# Model Configuration
EMBEDDING_MODEL = "text-embedding-3-small"
LLM_MODEL = "gpt-4o"
LLM_TEMPERATURE = 0.0 