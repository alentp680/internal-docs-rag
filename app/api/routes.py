
from pydantic import BaseModel, Field
from app.rag_pipeline import ask

from pathlib import Path
import shutil

from fastapi import APIRouter, UploadFile, File, HTTPException

from app.ingestion.ingestion_service import (
    ingest_document,
    delete_document_from_qdrant,
)

router = APIRouter()

UPLOAD_DIR = Path("data/raw")
ALLOWED_EXTENSIONS = {".txt", ".pdf", ".docx"}


@router.get("/health")
def health_check():
    return {
        "status": "healthy",
        "message": "RAG API routes are working"
    }


@router.post("/documents/upload")
def upload_document(file: UploadFile = File(...)):

    extension = Path(file.filename or "").suffix.lower()

    if extension not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail="Only TXT, PDF, and DOCX files are supported."
        )

    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

    file_path = UPLOAD_DIR / Path(file.filename).name

    try:
        with file_path.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        chunk_count = ingest_document(str(file_path))

        return {
            "message": "Document uploaded and indexed successfully.",
            "filename": file_path.name,
            "chunks_created": chunk_count
        }

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Document ingestion failed: {exc}"
        ) from exc

    finally:
        file.file.close()

@router.get("/documents")
def list_documents():
    from app.retrieval.document_index import document_index

    documents = document_index.documents

    return {
        "total_documents": len(documents),
        "documents": [
            {"filename": Path(file_path).name}
            for file_path in documents
        ]
    }


@router.delete("/documents/{filename}")
def delete_document(filename: str):
    from app.retrieval.document_index import document_index

    # Build the path to the requested file
    file_path = UPLOAD_DIR / Path(filename).name

    # Prevent invalid filenames
    if Path(filename).name != filename:
        raise HTTPException(
            status_code=400,
            detail="Invalid filename."
        )

    # Check whether the document is indexed
    if filename not in [
        Path(path).name for path in document_index.documents
    ]:
        raise HTTPException(
            status_code=404,
            detail="Document not found."
        )

    try:
        # Delete the document's Qdrant points
        delete_document_from_qdrant(filename)

        # Remove it from the in-memory index
        document_index.remove_document(filename)

        # Remove the original file if it exists
        if file_path.exists():
            file_path.unlink()

        return {
            "message": "Document deleted successfully.",
            "filename": filename
        }

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Document deletion failed: {exc}"
        ) from exc



class QueryRequest(BaseModel):
    question: str = Field(
        ...,
        min_length=1,
        description="Question to ask the RAG assistant",
    )
    top_k: int = Field(
        default=3,
        ge=1,
        le=10,
        description="Number of search results to retrieve",
    )


@router.post("/query")
def query_documents(request: QueryRequest):
    try:
        result = ask(
            query=request.question,
            top_k=request.top_k,
        )

        return {
            "question": request.question,
            "answer": result["answer"],
            "sources": result["sources"],
        }

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Question answering failed: {exc}",
        ) from exc
