from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, HTMLResponse
from fastapi.openapi.docs import get_swagger_ui_html
from typing import List, Dict, Any, Optional
from pydantic import BaseModel
import uuid
import os
import tempfile
import shutil
from pathlib import Path
import logging

from document_processor import DocumentProcessor, ALLOWED_FILE_TYPES, FILE_UPLOAD_LIMIT
from inference import InferenceAgent, LLMProvider
from search import call_search_engines
from claude import call_claude_llm_stream
from url_downloader import download_urls

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="AI Assistant API",
    description="API for processing documents, chatting with LLMs, and web search integration",
    version="1.0.0",
    docs_url=None,  # disable default /docs so we can serve a custom one
)

_ACCEPT = "." + ",.".join(ALLOWED_FILE_TYPES)

@app.get("/docs", include_in_schema=False)
async def custom_swagger_ui():
    html = get_swagger_ui_html(openapi_url="/openapi.json", title="AI Assistant API").body.decode()
    script = f"""
<script>
(function () {{
    const ACCEPT = "{_ACCEPT}";
    new MutationObserver(function () {{
        document.querySelectorAll('input[type="file"]').forEach(function (el) {{
            if (!el.getAttribute("accept")) {{
                el.setAttribute("accept", ACCEPT);
            }}
        }});
    }}).observe(document.body, {{ childList: true, subtree: true }});
}})();
</script>"""
    html = html.replace("</body>", script + "\n</body>")
    return HTMLResponse(content=html)

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

# Supported file extensions (must match DocumentProcessor)
SUPPORTED_EXTENSIONS = {
    'xlsx', 'xls', 'pdf', 'ppt', 'pptx', 'docx', 'doc',
    'txt', 'csv', 'html', 'htm', 'png', 'jpg', 'jpeg',
    'tiff', 'bmp', 'gif', 'md', 'markdown'
}


# ---------- Pydantic models ----------

class ChatRequest(BaseModel):
    session_id: str
    message: str


class UploadResponse(BaseModel):
    session_id: str
    document_count: int
    chunks_processed: int


class UrlUploadRequest(BaseModel):
    urls: str
    session_id: Optional[str] = None


class UrlUploadResponse(BaseModel):
    session_id: str
    document_count: int
    chunks_processed: int
    url_download_summary: Dict[str, Any]


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


def cleanup_temp_dir(dir_path: str):
    """Remove an entire temporary directory and its contents."""
    try:
        shutil.rmtree(dir_path, ignore_errors=True)
    except Exception as e:
        logger.error(
            f"Error removing temporary directory {dir_path}: {str(e)}")


def parse_urls(raw_urls: Optional[str]) -> List[str]:
    """Parse a comma-separated URL string into a validated list of URLs."""
    if not raw_urls or not raw_urls.strip():
        return []

    candidates = [u.strip() for u in raw_urls.split(",") if u.strip()]

    for url in candidates:
        if not url.startswith(("http://", "https://")):
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Invalid URL format: '{url}'. "
                    f"All URLs must start with http:// or https://"
                )
            )

    return candidates


class FileObject:
    """Wraps a local file path to mimic an uploaded file for DocumentProcessor."""

    def __init__(self, path: Path, name: str):
        self.path = path
        self.name = name

    def getbuffer(self) -> bytes:
        with open(self.path, "rb") as f:
            return f.read()


# ---------- Routes ----------

@app.get("/health")
async def health_check():
    return {"status": "ok"}


# =====================================================================
# ORIGINAL /upload endpoint — UNTOUCHED
# =====================================================================

@app.post("/upload", response_model=UploadResponse)
async def upload_documents(
    files: List[UploadFile] = File(...),
    session_id: Optional[str] = None,
    background_tasks: BackgroundTasks = None
):
    """Upload and process documents, creating a new session if needed."""
    if len(files) > FILE_UPLOAD_LIMIT:
        raise HTTPException(
            status_code=400,
            detail=f"Too many files: {len(files)} uploaded, maximum allowed is {FILE_UPLOAD_LIMIT}."
        )

    invalid_files = [
        file.filename for file in files
        if file.filename.split('.')[-1].lower() not in ALLOWED_FILE_TYPES
    ]
    if invalid_files:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type(s): {', '.join(invalid_files)}. Allowed formats: {', '.join(ALLOWED_FILE_TYPES)}."
        )

    if not session_id or session_id not in sessions:
        session_id = create_new_session()

    session = sessions[session_id]
    doc_processor = session["doc_processor"]

    temp_files: List[Path] = []
    uploaded_files = []

    for file in files:
        try:
            temp_path = create_temp_file(file)
            temp_files.append(temp_path)

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


# =====================================================================
# NEW /upload-url endpoint — URL-based document ingestion
# =====================================================================

@app.post("/upload-url", response_model=UrlUploadResponse)
async def upload_documents_from_url(
    request: UrlUploadRequest,
    background_tasks: BackgroundTasks,
):
    """
    Download and process documents from public URLs.

    Accepts a JSON body with a comma-separated list of URLs and an
    optional session_id. If session_id is provided and exists, documents
    are added to that session's vector store — so you can first call
    /upload with files, get back a session_id, then call /upload-url
    with that same session_id to add URL documents into the same session.

    Supported URL types:
      - Direct document links (PDF, DOCX, XLSX, PPTX, TXT, CSV, HTML,
        PNG, JPG, GIF, BMP, TIFF, MD, etc.)
      - ZIP archives (auto-extracted; all supported files inside are
        processed individually)
      - Google Drive file links:
          https://drive.google.com/file/d/FILE_ID/view
          https://drive.google.com/open?id=FILE_ID
      - Google Drive folder links:
          https://drive.google.com/drive/folders/FOLDER_ID
      - Dropbox sharing links (dl=0 auto-converted to dl=1)
      - OneDrive sharing links (download=1 auto-appended)
      - SharePoint sharing links (download=1 auto-appended)

    Example request body:
    {
        "urls": "https://example.com/report.pdf,https://drive.google.com/file/d/ABC/view",
        "session_id": "optional-existing-session-id"
    }
    """
    # ── Validate URLs ────────────────────────────────────────────────────
    parsed_urls = parse_urls(request.urls)

    if not parsed_urls:
        raise HTTPException(
            status_code=400,
            detail="No valid URLs provided. Supply at least one URL."
        )

    # ── Session setup (reuse existing or create new) ─────────────────────
    session_id = request.session_id
    if not session_id or session_id not in sessions:
        session_id = create_new_session()

    session = sessions[session_id]
    doc_processor = session["doc_processor"]

    # ── Download files from URLs ─────────────────────────────────────────
    url_download_temp_dir = tempfile.mkdtemp(prefix="url_dl_")

    url_summary: Dict[str, Any] = {
        "total_requested": len(parsed_urls),
        "total_files_downloaded": 0,
        "failed_urls": [],
        "downloaded_files": [],
    }

    try:
        local_paths = await download_urls(
            urls=parsed_urls,
            target_dir=url_download_temp_dir
        )
    except Exception as e:
        logger.error(f"Unexpected error from download_urls: {e}")
        local_paths = []

    # ── Wrap downloaded files as FileObjects ──────────────────────────────
    url_file_objects: List[FileObject] = []

    for local_path in local_paths:
        basename = os.path.basename(local_path)
        file_ext = os.path.splitext(basename)[1].lstrip('.').lower()

        if file_ext not in SUPPORTED_EXTENSIONS:
            logger.warning(
                f"URL-downloaded file has unsupported extension, "
                f"skipping: {basename}"
            )
            continue

        url_file_objects.append(FileObject(Path(local_path), basename))
        url_summary["downloaded_files"].append(basename)

    url_summary["total_files_downloaded"] = len(url_file_objects)

    if len(local_paths) == 0 and len(parsed_urls) > 0:
        url_summary["failed_urls"] = parsed_urls
    elif len(local_paths) < len(parsed_urls):
        url_summary["note"] = (
            f"{len(parsed_urls)} URL(s) requested, "
            f"{len(local_paths)} file(s) downloaded. "
            f"Some URLs may have failed — check server logs for details."
        )

    if not url_file_objects:
        background_tasks.add_task(cleanup_temp_dir, url_download_temp_dir)
        raise HTTPException(
            status_code=400,
            detail=(
                "No processable files found. All URL downloads may have "
                "failed or returned unsupported file types. "
                "Check server logs for per-URL error details."
            )
        )

    # ── Process documents ────────────────────────────────────────────────
    try:
        chunks_processed = doc_processor.process_documents(url_file_objects)
        session["document_count"] += len(url_file_objects)

        background_tasks.add_task(cleanup_temp_dir, url_download_temp_dir)

        return UrlUploadResponse(
            session_id=session_id,
            document_count=session["document_count"],
            chunks_processed=chunks_processed,
            url_download_summary=url_summary,
        )

    except Exception as e:
        cleanup_temp_dir(url_download_temp_dir)
        raise HTTPException(
            status_code=500,
            detail=f"Error processing documents: {str(e)}"
        )


# =====================================================================
# Chat & Search endpoints — UNTOUCHED
# =====================================================================

@app.post("/deepthink")
async def chat_with_claude(request: ChatRequest):
    """Chat with Claude Sonnet 4.5 (with Mistral fallback) using a specific session with extended thinking and streaming."""
    session = get_session(request.session_id)

    session["chat_messages"].append(
        {"role": "user", "content": request.message})

    context = None
    if session["doc_processor"].get_vector_store() is not None:
        context = session["doc_processor"].search_documents(
            request.message, k=3)

    llm_params = session["inference_agent"].generate_chat_response(
        session["chat_messages"],
        context=context,
        provider=LLMProvider.CLAUDE
    )

    async def format_stream():
        full_response = ""

        # Check if we're using Claude (has thinking) or Mistral (no thinking)
        is_claude = "thinking" in llm_params

        if is_claude:
            # Claude streaming with extended thinking
            current_thinking = ""
            in_thinking_block = False
            in_text_block = False

            with llm_params["client"].messages.stream(
                model=llm_params["model"],
                max_tokens=llm_params["max_tokens"],
                thinking=llm_params["thinking"],
                messages=llm_params["messages"]
            ) as stream:
                for chunk in stream:
                    chunk_str = str(chunk)

                    # Parse thinking block
                    if "content_block_delta" in chunk_str and "thinking_delta" in chunk_str:
                        if not in_thinking_block:
                            in_thinking_block = True
                            yield "<think>\n"

                        if hasattr(chunk, 'delta') and hasattr(chunk.delta, 'thinking'):
                            thinking_content = chunk.delta.thinking
                            current_thinking += thinking_content
                            yield thinking_content

                    # Detect end of thinking block
                    elif "content_block_stop" in chunk_str and "thinking" in chunk_str and in_thinking_block:
                        in_thinking_block = False
                        yield "\n</think>\n\n"

                    # Parse text block (final answer)
                    elif "content_block_delta" in chunk_str and "text_delta" in chunk_str:
                        if not in_text_block:
                            in_text_block = True
                            yield "<answer>\n"

                        if hasattr(chunk, 'delta') and hasattr(chunk.delta, 'text'):
                            text_content = chunk.delta.text
                            full_response += text_content
                            yield text_content

                    # Detect end of text block
                    elif "content_block_stop" in chunk_str and "text" in chunk_str and in_text_block:
                        in_text_block = False
                        yield "\n</answer>"
        else:
            # Mistral streaming (no extended thinking)
            logger.info("Using Mistral fallback for chat completion")

            stream = llm_params["client"].chat.stream(
                model=llm_params["model"],
                messages=llm_params["messages"],
                max_tokens=llm_params["max_tokens"],
                temperature=llm_params.get("temperature", 0.7)
            )

            for chunk in stream:
                if hasattr(chunk, 'data') and chunk.data:
                    if hasattr(chunk.data, 'choices') and chunk.data.choices:
                        delta = chunk.data.choices[0].delta
                        if hasattr(delta, 'content') and delta.content:
                            full_response += delta.content
                            yield delta.content

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
