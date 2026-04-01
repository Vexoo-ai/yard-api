from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from typing import List, Dict, Any, Optional
from pydantic import BaseModel
import uuid
import os
import tempfile
import shutil
from pathlib import Path
import logging

from document_processor import DocumentProcessor
from inference import InferenceAgent, LLMProvider
from search import call_search_engines
from claude import call_claude_llm_stream

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="AI Assistant API",
    description="API for processing documents, chatting with LLMs, and web search integration",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory session store
sessions: Dict[str, Dict[str, Any]] = {}


# ---------- Pydantic models ----------

class ChatRequest(BaseModel):
    session_id: str
    message: str


class UploadResponse(BaseModel):
    session_id: str
    document_count: int
    chunks_processed: int


class SerpInput(BaseModel):
    query: str


class SerpRequest(BaseModel):
    input: SerpInput


# ---------- Helpers ----------

def get_session(session_id: str) -> Dict[str, Any]:
    if session_id not in sessions:
        raise HTTPException(
            status_code=404, detail=f"Session {session_id} not found")
    return sessions[session_id]


def create_new_session() -> str:
    session_id = str(uuid.uuid4())
    sessions[session_id] = {
        "doc_processor": DocumentProcessor(),
        "inference_agent": InferenceAgent(),
        "chat_messages": [],
        "document_count": 0
    }
    return session_id


def create_temp_file(file: UploadFile) -> Path:
    suffix = os.path.splitext(file.filename)[1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        temp_path = Path(tmp.name)
    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    return temp_path


def cleanup_temp_files(files: List[Path]):
    for file_path in files:
        try:
            os.unlink(file_path)
        except Exception as e:
            logger.error(
                f"Error removing temporary file {file_path}: {str(e)}")


# ---------- Routes ----------

@app.get("/health")
async def health_check():
    return {"status": "ok"}


@app.post("/upload", response_model=UploadResponse)
async def upload_documents(
    files: List[UploadFile] = File(...),
    session_id: Optional[str] = None,
    background_tasks: BackgroundTasks = None
):
    """Upload and process documents, creating a new session if needed."""
    if not session_id or session_id not in sessions:
        session_id = create_new_session()

    session = sessions[session_id]
    doc_processor = session["doc_processor"]

    temp_files: List[Path] = []
    uploaded_files = []

    for file in files:
        try:
            file_type = file.filename.split('.')[-1].lower()
            if file_type not in ['xlsx', 'xls', 'pdf', 'ppt', 'pptx', 'docx', 'doc',
                                 'txt', 'csv', 'html', 'htm', 'png', 'jpg', 'jpeg',
                                 'tiff', 'bmp', 'gif']:
                logger.warning(f"Skipping unsupported file type: {file_type}")
                continue

            temp_path = create_temp_file(file)
            temp_files.append(temp_path)

            class FileObject:
                def __init__(self, path: Path, name: str):
                    self.path = path
                    self.name = name

                def getbuffer(self):
                    with open(self.path, "rb") as f:
                        return f.read()

            uploaded_files.append(FileObject(temp_path, file.filename))

        except Exception as e:
            cleanup_temp_files(temp_files)
            raise HTTPException(
                status_code=400,
                detail=f"Error processing file {file.filename}: {str(e)}"
            )

    try:
        chunks_processed = doc_processor.process_documents(uploaded_files)
        session["document_count"] += len(uploaded_files)

        if background_tasks:
            background_tasks.add_task(cleanup_temp_files, temp_files)
        else:
            cleanup_temp_files(temp_files)

        return UploadResponse(
            session_id=session_id,
            document_count=session["document_count"],
            chunks_processed=chunks_processed
        )
    except Exception as e:
        cleanup_temp_files(temp_files)
        raise HTTPException(
            status_code=500, detail=f"Error processing documents: {str(e)}")


@app.post("/deepthink")
async def chat_with_claude(request: ChatRequest):
    """Chat with Claude Sonnet 4.6 using a specific session with streaming."""
    session = get_session(request.session_id)

    session["chat_messages"].append(
        {"role": "user", "content": request.message})

    context = None
    if session["doc_processor"].get_vector_store() is not None:
        context = session["doc_processor"].search_documents(
            request.message, k=3)

    claude_params = session["inference_agent"].generate_chat_response(
        session["chat_messages"],
        context=context,
        provider=LLMProvider.CLAUDE
    )

    async def format_stream():
        full_response = ""

        with claude_params["client"].messages.stream(
            model=claude_params["model"],
            max_tokens=claude_params["max_tokens"],
            messages=claude_params["messages"]
        ) as stream:
            for text in stream.text_stream:
                full_response += text
                yield text

        session["chat_messages"].append(
            {"role": "assistant", "content": full_response})

    return StreamingResponse(format_stream(), media_type="text/plain")


@app.post("/atlas")
async def get_atlas_response(request: SerpRequest) -> StreamingResponse:
    """Perform web search and stream Claude AI response."""
    args = request.input
    if not args or not args.query:
        raise HTTPException(
            status_code=400, detail="Invalid input: query is required")

    try:
        logger.info(f"Processing atlas request for query: {args.query}")
        search_results = await call_search_engines(args.query)
        logger.info(
            f"Search results retrieved: {len(search_results.get('organic_results', []))} results")

        async def generate():
            async for chunk in call_claude_llm_stream(args.query, search_results):
                if chunk is not None:
                    yield str(chunk)

        return StreamingResponse(generate(), media_type="text/plain")

    except Exception as e:
        logger.error(f"Error in atlas route: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500, detail=f"An error occurred: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8010)
