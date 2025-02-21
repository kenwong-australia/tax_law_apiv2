from fastapi import HTTPException
from pydantic import BaseModel
import requests
from docx import Document
import tempfile
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DocumentURL(BaseModel):
    url: str
    flow_variable: str  # Name of the flutterflow variable to store text in

def process_docx(file_path: str) -> str:
    """Extract text from a .docx file"""
    try:
        doc = Document(file_path)
        full_text = []
        for paragraph in doc.paragraphs:
            full_text.append(paragraph.text)
        return '\n'.join(full_text)
    except Exception as e:
        logger.error(f"Error processing document: {str(e)}")
        raise

async def process_document_url(doc_request: DocumentURL):
    try:
        # Download the file from Firestore URL
        response = requests.get(doc_request.url)
        response.raise_for_status()

        # Create a temporary file to store the document
        with tempfile.NamedTemporaryFile(delete=False, suffix='.docx') as temp_file:
            temp_file.write(response.content)
            temp_path = temp_file.name

        try:
            # Process the document
            document_text = process_docx(temp_path)
            
            return {
                "status": "success",
                "variable_name": doc_request.flow_variable,
                "text": document_text,
                "message": f"Document processed and stored in {doc_request.flow_variable}"
            }
        finally:
            # Clean up the temporary file
            if os.path.exists(temp_path):
                os.remove(temp_path)

    except requests.RequestException as e:
        logger.error(f"Error downloading document: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Error downloading document: {str(e)}")
    except Exception as e:
        logger.error(f"Error processing document: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing document: {str(e)}") 