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

# Import your existing modules
from document_processor import DocumentProcessor
from inference import InferenceAgent, LLMProvider
from search import call_search_engines
from mistral import call_mistral_llm_stream

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="AI Assistant API",
    description="API for processing documents, chatting with LLMs, and web search integration",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Session management
sessions: Dict[str, Dict[str, Any]] = {}

# Define request and response models
class ChatRequest(BaseModel):
    session_id: str
    message: str

class UploadResponse(BaseModel):
    session_id: str
    document_count: int
    chunks_processed: int

class SerpRequest(BaseModel):
    class Input(BaseModel):
        query: str
    
    input: Input

# Helper functions
def get_session(session_id: str):
    """Get session by ID or raise an exception if not found"""
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found")
    return sessions[session_id]

def create_temp_file(file: UploadFile) -> Path:
    """Create a temporary file from an uploaded file"""
    # Create a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as temp_file:
        # Get the file path
        temp_path = Path(temp_file.name)
        
    # Write uploaded file content to the temporary file
    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        
    return temp_path

def cleanup_temp_files(files: List[Path]):
    """Clean up temporary files"""
    for file_path in files:
        try:
            os.unlink(file_path)
        except Exception as e:
            logger.error(f"Error removing temporary file {file_path}: {str(e)}")

@app.post("/upload", response_model=UploadResponse)
async def upload_documents(
    files: List[UploadFile] = File(...),
    session_id: Optional[str] = None,
    background_tasks: BackgroundTasks = None
):
    """Upload and process documents, creating a new session if needed"""
    # Create a new session if not provided
    if not session_id or session_id not in sessions:
        session_id = str(uuid.uuid4())
        sessions[session_id] = {
            "doc_processor": DocumentProcessor(),
            "inference_agent": InferenceAgent(),
            "chat_messages": [],
            "document_count": 0
        }
    
    session = sessions[session_id]
    doc_processor = session["doc_processor"]
    
    # Create temporary files for processing
    temp_files = []
    uploaded_files = []
    
    # Save uploaded files to temporary location
    for file in files:
        try:
            # Check file type
            file_type = file.filename.split('.')[-1].lower()
            if file_type not in ['xlsx', 'xls', 'pdf', 'ppt', 'pptx', 'docx', 'doc', 'txt', 'csv', 'html', 'htm']:
                continue
                
            # Create temp file
            temp_path = create_temp_file(file)
            temp_files.append(temp_path)
            
            # Create file-like object that our DocumentProcessor can use
            class FileObject:
                def __init__(self, path, name):
                    self.path = path
                    self.name = name
                    
                def getbuffer(self):
                    with open(self.path, "rb") as f:
                        return f.read()
            
            uploaded_files.append(FileObject(temp_path, file.filename))
        except Exception as e:
            # Clean up files on error
            cleanup_temp_files(temp_files)
            raise HTTPException(status_code=400, detail=f"Error processing file {file.filename}: {str(e)}")
    
    # Process documents
    try:
        chunks_processed = doc_processor.process_documents(uploaded_files)
        session["document_count"] += len(uploaded_files)
        
        # Schedule cleanup of temporary files
        if background_tasks:
            background_tasks.add_task(cleanup_temp_files, temp_files)
        
        return UploadResponse(
            session_id=session_id,
            document_count=session["document_count"],
            chunks_processed=chunks_processed
        )
    except Exception as e:
        # Clean up files on error
        cleanup_temp_files(temp_files)
        raise HTTPException(status_code=500, detail=f"Error processing documents: {str(e)}")

@app.post("/deepthink")
async def chat_with_claude(request: ChatRequest):
    """Chat with Claude using a specific session with formatted streaming response"""
    session = get_session(request.session_id)
    
    # Add user message to chat history
    session["chat_messages"].append({"role": "user", "content": request.message})
    
    # Get relevant context from vector store
    context = None
    if session["doc_processor"].get_vector_store() is not None:
        context = session["doc_processor"].search_documents(request.message, k=3)
    
    # Get Claude parameters
    claude_params = session["inference_agent"].generate_chat_response(
        session["chat_messages"],
        context=context,
        provider=LLMProvider.CLAUDE
    )
    
    async def format_stream():
        full_response = ""
        current_thinking = ""
        in_thinking_block = False
        in_text_block = False
        
        # Create the stream and process events
        with claude_params["client"].messages.stream(
            model=claude_params["model"],
            max_tokens=claude_params["max_tokens"],
            thinking=claude_params["thinking"],
            messages=claude_params["messages"]
        ) as stream:
            for chunk in stream:
                chunk_str = str(chunk)
                
                # Parse thinking block
                if "content_block_delta" in chunk_str and "thinking_delta" in chunk_str:
                    # This is a thinking delta
                    if not in_thinking_block:
                        in_thinking_block = True
                        yield "<think>\n"
                    
                    # Extract thinking content from ThinkingEvent
                    if hasattr(chunk, 'delta') and hasattr(chunk.delta, 'thinking'):
                        thinking_content = chunk.delta.thinking
                        current_thinking += thinking_content
                        yield thinking_content
                
                # Detect the end of thinking block
                elif "content_block_stop" in chunk_str and "thinking" in chunk_str and in_thinking_block:
                    in_thinking_block = False
                    yield "\n</think>\n\n"
                
                # Parse text block
                elif "content_block_delta" in chunk_str and "text_delta" in chunk_str:
                    # This is a text delta (final answer)
                    if not in_text_block:
                        in_text_block = True
                        yield "<answer>\n"
                    
                    # Extract text content
                    if hasattr(chunk, 'delta') and hasattr(chunk.delta, 'text'):
                        text_content = chunk.delta.text
                        full_response += text_content
                        yield text_content
                
                # Detect the end of text block
                elif "content_block_stop" in chunk_str and "text" in chunk_str and in_text_block:
                    in_text_block = False
                    yield "\n</answer>"
        
        # Add assistant response to chat history
        session["chat_messages"].append({"role": "assistant", "content": full_response})
    
    return StreamingResponse(format_stream(), media_type="text/plain")

@app.post("/atlas")
async def get_mistral_response(request: SerpRequest) -> StreamingResponse:
    """Process search results and stream Mistral LLM responses"""
    args = request.input
    if not args or not args.query:
        raise HTTPException(status_code=400, detail="Invalid input: query is required")
    
    try:
        # Log the incoming request
        logger.info(f"Processing Mistral request for query: {args.query}")
        
        # Call search engines to get results
        search_results = await call_search_engines(args.query)
        logger.info(f"Search results retrieved: {len(search_results.get('organic_results', []))} results")
        
        # Create streaming response
        async def generate():
            # Get the generator from call_mistral_llm_stream
            mistral_generator = call_mistral_llm_stream(args.query, search_results)
            
            # Correctly iterate through an async generator
            async for chunk in mistral_generator:
                if chunk is not None:
                    yield str(chunk)
                else:
                    logger.warning("Received None chunk from Mistral LLM stream")
        
        return StreamingResponse(generate(), media_type="text/plain")
    
    except Exception as e:
        logger.error(f"Error in Mistral API route: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)