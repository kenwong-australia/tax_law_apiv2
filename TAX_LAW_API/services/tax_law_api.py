from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, Dict, Any, List
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from langchain_openai import ChatOpenAI
import os
import logging
import requests
from docx import Document
import tempfile

from TAX_LAW_API.utils.config import (
    OPENAI_API_KEY, 
    PINECONE_API_KEY, 
    PINECONE_INDEX_NAME,
    LLM_MODEL,
    LLM_TEMPERATURE
)
from TAX_LAW_API.utils.embeddings import create_vector_store
from .document_service import DocumentURL, process_document_url

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

    def generate_response_prompt(self, query_params: TaxQuery, context: str) -> str:
        return f"""You are a tax law advisor in Australia. Analyze this query and provide exactly six responses.

Query: {query_params.query}

Context: {context}

Respond in exactly this format with these exact section headers:

[TITLE]
{query_params.title}

[TAX_RESEARCH]
{query_params.tax_research}

[TAX_CITATIONS]
{query_params.tax_citations}

[DRAFT_CLIENT_RESPONSE]
{query_params.draft_client_response}

[CLARIFYING_QUESTIONS]
{query_params.clarifying_questions}

[CONFIRMATION]
{query_params.confirmation}"""

    def extract_citations(self, citations_text: str) -> List[Dict[str, str]]:
        """Extract citations from text formatted as "Citation Name | Citation URL" """
        citations = []
        try:
            for line in citations_text.strip().split('\n'):
                if '|' in line:
                    parts = line.split('|', 1)
                    if len(parts) == 2:
                        citations.append({
                            "citations_name": parts[0].strip(),
                            "citation_url": parts[1].strip()
                        })
        except Exception as e:
            logger.error(f"Error extracting citations: {str(e)}")
        return citations or []  # Return empty list if no citations found

    def parse_response(self, response: str) -> Dict[str, Any]:
        # Initialize with default values
        sections = {
            "title": "Untitled",
            "tax_research": "No research provided",
            "tax_citations": "No citations provided",
            "draft_client_response": "No draft provided",
            "clarifying_questions": "No questions provided",
            "confirmation": "No confirmation provided"
        }
        
        try:
            # Map section headers to JSON keys
            header_to_key = {
                "[TITLE]": "title",
                "[TAX_RESEARCH]": "tax_research",
                "[TAX_CITATIONS]": "tax_citations",
                "[DRAFT_CLIENT_RESPONSE]": "draft_client_response",
                "[CLARIFYING_QUESTIONS]": "clarifying_questions",
                "[CONFIRMATION]": "confirmation"
            }
            
            current_section = None
            current_content = []
            
            for line in response.split('\n'):
                line_stripped = line.strip()
                if line_stripped in header_to_key:
                    if current_section and current_section in header_to_key:
                        sections[header_to_key[current_section]] = '\n'.join(current_content).strip()
                    current_section = line_stripped
                    current_content = []
                elif current_section and line.strip():
                    current_content.append(line)
            
            # Save the last section
            if current_section and current_section in header_to_key and current_content:
                sections[header_to_key[current_section]] = '\n'.join(current_content).strip()
            
            # Extract citations
            citations = self.extract_citations(sections["tax_citations"])
            
            return sections, citations
            
        except Exception as e:
            logger.error(f"Error parsing response: {str(e)}")
            return sections, []

    def answer_question(self, query_params: TaxQuery) -> Dict[str, Any]:
        try:
            # Retrieve context
            context = self.query_engine.retrieve_context(query_params.query)
            
            # Generate response
            prompt = self.generate_response_prompt(query_params, context)
            response = self.llm.invoke(prompt)
            
            # Parse response
            sections, citations = self.parse_response(response.content)
            
            # Create the response structure
            result = sections.copy()
            result["citations"] = citations
            
            return result
            
        except Exception as e:
            logger.error(f"Error in answer_question: {str(e)}")
            # Return a fallback response
            return {
                "title": "Error Processing Query",
                "tax_research": "An error occurred while processing your query.",
                "tax_citations": "No citations available",
                "draft_client_response": "Unable to generate response at this time.",
                "clarifying_questions": "Service temporarily unavailable.",
                "confirmation": "Error occurred during processing.",
                "citations": []
            }

# Global RAG instance
rag_instance = None

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@app.on_event("startup")
async def startup_event():
    global rag_instance
    logger.info("Starting up Tax Law API...")
    if not all([OPENAI_API_KEY, PINECONE_API_KEY]):
        logger.error("Missing required environment variables")
        raise RuntimeError("Missing required environment variables")
    
    try:
        logger.info("Initializing RAG system...")
        query_engine = TaxLawQueryEngine()
        rag_instance = TaxLawRAG(query_engine)
        logger.info("RAG system initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize RAG system: {str(e)}")
        raise

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

@app.get("/")
async def root():
    return {"message": "Tax Law API is running"}

@app.post("/process-document")
async def process_document(doc_request: DocumentURL):
    return await process_document_url(doc_request)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "8000"))) 