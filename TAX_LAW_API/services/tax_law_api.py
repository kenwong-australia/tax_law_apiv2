from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, Dict, Any, List
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from langchain_openai import ChatOpenAI

from utils.config import (
    OPENAI_API_KEY, 
    PINECONE_API_KEY, 
    PINECONE_INDEX_NAME,
    LLM_MODEL,
    LLM_TEMPERATURE
)
from utils.embeddings import create_vector_store

app = FastAPI(title="Tax Law RAG API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class TaxQuery(BaseModel):
    query: str
    title: str
    tax_research: str
    tax_citations: str
    draft_client_response: str
    clarifying_questions: str
    confirmation: str

class TaxLawQueryEngine:
    def __init__(self):
        self.vector_store = create_vector_store(PINECONE_INDEX_NAME)

    def retrieve_context(self, query: str, k: int = 4) -> str:
        results = self.vector_store.similarity_search(query, k=k)
        context = ""
        for result in results:
            context += f"\nSection: {result.metadata['full_reference']}\n"
            context += f"{result.page_content}\n"
        return context

class TaxLawRAG:
    def __init__(self, query_engine: TaxLawQueryEngine):
        self.query_engine = query_engine
        self.llm = ChatOpenAI(
            openai_api_key=OPENAI_API_KEY,
            model=LLM_MODEL,
            temperature=LLM_TEMPERATURE
        )

    # [Rest of the TaxLawRAG class methods remain the same]
    def generate_response_prompt(self, query_params: TaxQuery, context: str) -> str:
        # [Existing method implementation]
        pass

    def extract_citations(self, citations_text: str) -> List[Dict[str, str]]:
        # [Existing method implementation]
        pass

    def parse_response(self, response: str) -> Dict[str, Any]:
        # [Existing method implementation]
        pass

    def answer_question(self, query_params: TaxQuery) -> Dict[str, Any]:
        # [Existing method implementation]
        pass

# Global RAG instance
rag_instance = None

@app.on_event("startup")
async def startup_event():
    global rag_instance
    if not all([OPENAI_API_KEY, PINECONE_API_KEY]):
        raise RuntimeError("Missing required environment variables")
    
    print("Initializing RAG system...")
    query_engine = TaxLawQueryEngine()
    rag_instance = TaxLawRAG(query_engine)
    print("RAG system initialized successfully")

@app.post("/query")
async def query_tax_law(query_params: TaxQuery):
    if not rag_instance:
        raise HTTPException(status_code=500, detail="RAG system not initialized")
    try:
        result = rag_instance.answer_question(query_params)
        if "citations" not in result or not isinstance(result["citations"], list):
            result["citations"] = []
        return JSONResponse(content=result)
    except Exception as e:
        print(f"Error processing query: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "8000"))) 