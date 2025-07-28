from fastapi import FastAPI, UploadFile, File, Form, HTTPException, BackgroundTasks, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.exceptions import RequestValidationError
from pydantic import BaseModel, ValidationError
import os
import shutil
from dotenv import load_dotenv
from typing import Dict, List, Optional, AsyncGenerator
import uuid
import logging
from enum import Enum
import json
import asyncio
import nltk
import time
from functools import wraps

from llama_index.core.chat_engine.types import StreamingAgentChatResponse

# LlamaIndex imports
from llama_index.core import (
    Settings,
    StorageContext,
    KnowledgeGraphIndex,
    SimpleDirectoryReader,
    load_index_from_storage,
)
try:
    from llama_index.readers.file import PDFReader
except ImportError:
    logger.warning("PDFReader not available, using default PDF parsing")
    PDFReader = None

# Initialize logger first to avoid NameError
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("main")

# Try to import Langfuse decorator for better integration
try:
    from langfuse.decorators import observe
    langfuse_observe_available = True
    logger.info("Langfuse observe decorator available")
except ImportError:
    langfuse_observe_available = False
    logger.info("Langfuse observe decorator not available")
from llama_index.core.callbacks import CallbackManager, CBEventType, EventPayload
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.graph_stores.neo4j import Neo4jGraphStore

# Load environment variables
load_dotenv()

# --- Additional Logging Setup ---
# logging.getLogger('llama_index.core').setLevel(logging.DEBUG)

# --- Constants ---
UPLOAD_DIRECTORY = "./uploaded_files"
STORAGE_DIRECTORY = "./storage" # *** NEW: Directory to store index metadata ***
os.makedirs(UPLOAD_DIRECTORY, exist_ok=True)
os.makedirs(STORAGE_DIRECTORY, exist_ok=True)
CHATBOTS_METADATA_FILE = "./chatbots_metadata.json"

# Security settings for file uploads
MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB limit
ALLOWED_EXTENSIONS = {'.pdf', '.txt', '.md', '.docx', '.doc'}
ALLOWED_MIME_TYPES = {
    'application/pdf',
    'text/plain',
    'text/markdown',
    'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
    'application/msword'
}

# --- FastAPI App Initialization ---
app = FastAPI(
    title="GraphLM Backend",
    description="API for creating and managing GraphRAG-powered chatbots.",
    version="0.4.0", # Version bump to reflect improvements
)

# Security improvement: Restrict CORS to specific origins
ALLOWED_ORIGINS = [
    "http://localhost:3000",  # Frontend development
    "http://127.0.0.1:3000",  # Alternative localhost
]

# Add production origins from environment variable if available
if production_origins := os.getenv("ALLOWED_ORIGINS"):
    ALLOWED_ORIGINS.extend(production_origins.split(","))

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization", "Accept"],
)

app.mount("/files", StaticFiles(directory=UPLOAD_DIRECTORY), name="files")

# Global exception handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Handle HTTP exceptions with detailed error messages."""
    logger.error(f"HTTP {exc.status_code} error at {request.url}: {exc.detail}")
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": "Request Error",
            "message": exc.detail,
            "status_code": exc.status_code,
            "timestamp": time.time()
        }
    )

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Handle request validation errors with user-friendly messages."""
    logger.error(f"Validation error at {request.url}: {exc.errors()}")
    return JSONResponse(
        status_code=422,
        content={
            "error": "Validation Error",
            "message": "Invalid request data provided",
            "details": exc.errors(),
            "timestamp": time.time()
        }
    )

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle unexpected errors with generic error message."""
    logger.error(f"Unexpected error at {request.url}: {str(exc)}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal Server Error",
            "message": "An unexpected error occurred. Please try again later.",
            "timestamp": time.time()
        }
    )

# Retry decorator for database operations
def retry_on_failure(max_attempts: int = 3, delay: float = 1.0):
    """Decorator to retry functions on failure with exponential backoff."""
    def decorator(func):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            last_exception = None
            for attempt in range(max_attempts):
                try:
                    if asyncio.iscoroutinefunction(func):
                        return await func(*args, **kwargs)
                    else:
                        return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    if attempt < max_attempts - 1:
                        wait_time = delay * (2 ** attempt)  # Exponential backoff
                        logger.warning(f"Attempt {attempt + 1} failed: {e}. Retrying in {wait_time}s...")
                        await asyncio.sleep(wait_time)
                    else:
                        logger.error(f"All {max_attempts} attempts failed: {e}")
            raise last_exception
        return async_wrapper
    return decorator

# --- Status Enums & API Models (No changes) ---
class ChatbotStatus(str, Enum):
    INDEXING = "INDEXING"
    READY = "READY"
    FAILED = "FAILED"

class IndexingStep(str, Enum):
    LOADING_DOCUMENTS = "Loading Documents"
    PARSING_NODES = "Parsing Nodes"
    GENERATING_EMBEDDINGS = "Generating Embeddings"
    BUILDING_GRAPH = "Building Graph"

class Chatbot(BaseModel):
    id: str
    name: str
    description: Optional[str] = None
    status: ChatbotStatus = ChatbotStatus.READY
    total_nodes: Optional[int] = 0
    processed_nodes: Optional[int] = 0
    current_step: Optional[IndexingStep] = None

class ChatbotUpdate(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None

class ChatRequest(BaseModel):
    chatbot_id: str
    query: str

class Source(BaseModel):
    document_name: str
    page_number: Optional[int] = None
    snippet: Optional[str] = None
    url: Optional[str] = None

class StreamEvent(str, Enum):
    MESSAGE = "message"
    SOURCES = "sources"
    GRAPH = "graph"
    DONE = "done"

class ChatResponse(BaseModel):
    event: StreamEvent
    data: Dict

class IndexingProgressResponse(BaseModel):
    total_nodes: int
    processed_nodes: int
    status: ChatbotStatus
    current_step: Optional[IndexingStep] = None

# --- In-Memory Stores ---
chatbots_db: Dict[str, Chatbot] = {}
query_engines: Dict[str, any] = {}

# --- LlamaIndex Global Settings ---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")

if not all([OPENAI_API_KEY, NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD]):
    raise ValueError("One or more required environment variables are not set.")

Settings.llm = OpenAI(temperature=0, model="gpt-4.1-nano-2025-04-14", api_key=OPENAI_API_KEY)
Settings.embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
Settings.num_workers = 4
Settings.chunk_size = 1024  # Optimal chunk size for most documents
Settings.chunk_overlap = 100  # Overlap between chunks for better context



from llama_index.core import get_response_synthesizer
from llama_index.core.prompts import PromptTemplate

# Define the custom QA template
QA_TEMPLATE_WITH_CITATION = PromptTemplate(
    """Context information is below.
---------------------
{context_str}
---------------------
Given the context information and not prior knowledge, answer the query in a comprehensive and accurate manner.

Instructions:
1. Analyze the context carefully to understand the relationships between different pieces of information.
2. Provide a detailed answer that synthesizes information from multiple sources when relevant.
3. When you use information from a source document, cite it by including the document's filename in parentheses at the end of the sentence or phrase. For example: "The capital of France is Paris (document.pdf)."
4. If you use information from multiple source documents for a single statement, cite all relevant filenames. For example: "The project was completed on time (report.pdf, minutes.docx)."
5. If the context doesn't contain sufficient information to answer the query, clearly state what information is missing.
6. If you don't use any source, do not include any citation.
7. Structure your answer logically and provide specific details when available.

Query: {query_str}
Answer: """
)

# Langfuse Integration
LANGFUSE_PUBLIC_KEY = os.getenv("LANGFUSE_PUBLIC_KEY")
LANGFUSE_SECRET_KEY = os.getenv("LANGFUSE_SECRET_KEY")
LANGFUSE_HOST = os.getenv("LANGFUSE_HOST")

# Initialize Langfuse client
langfuse_client = None
if all([LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY, LANGFUSE_HOST]):
    try:
        from langfuse import Langfuse
        langfuse_client = Langfuse(
            public_key=LANGFUSE_PUBLIC_KEY,
            secret_key=LANGFUSE_SECRET_KEY,
            host=LANGFUSE_HOST
        )
        logger.info("Langfuse client initialized successfully.")
    except ImportError:
        logger.error("Langfuse library not installed. Please install it with: pip install langfuse")
    except Exception as e:
        logger.error(f"Failed to initialize Langfuse client: {e}")
        langfuse_client = None
else:
    logger.warning("Langfuse environment variables not fully set. Skipping Langfuse integration.")

# Security utility functions
def validate_uploaded_file(file: UploadFile) -> None:
    """Validate uploaded file for security and size constraints."""
    # Check file size
    if hasattr(file, 'size') and file.size and file.size > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Maximum size allowed: {MAX_FILE_SIZE // (1024 * 1024)}MB"
        )
    
    # Check file extension
    if file.filename:
        file_ext = os.path.splitext(file.filename)[1].lower()
        if file_ext not in ALLOWED_EXTENSIONS:
            raise HTTPException(
                status_code=415,
                detail=f"File type not supported. Allowed extensions: {', '.join(ALLOWED_EXTENSIONS)}"
            )
    
    # Check MIME type
    if file.content_type and file.content_type not in ALLOWED_MIME_TYPES:
        raise HTTPException(
            status_code=415,
            detail=f"MIME type not supported. File type: {file.content_type}"
        )

def sanitize_filename(filename: str) -> str:
    """Sanitize filename to prevent directory traversal attacks."""
    import re
    # Remove any path separators and keep only alphanumeric, dots, hyphens, underscores
    sanitized = re.sub(r'[^\w\-_\.]', '_', filename)
    # Prevent hidden files and relative paths
    sanitized = sanitized.lstrip('.')
    return sanitized or 'uploaded_file'

# Utility function to log events to Langfuse
def log_to_langfuse(event_name: str, metadata: dict, tags: list = None):
    """Utility function to safely log events to Langfuse using the correct API."""
    if langfuse_client:
        try:
            # Debug: Print available methods on first call
            available_methods = [method for method in dir(langfuse_client) if not method.startswith('_')]
            logger.debug(f"Available Langfuse methods: {available_methods}")
            
            # Use the correct create_event method
            if hasattr(langfuse_client, 'create_event'):
                try:
                    # Add tags to metadata if provided
                    enhanced_metadata = metadata.copy()
                    if tags:
                        enhanced_metadata['tags'] = tags
                    
                    # For chat queries, try to create a more structured trace
                    if 'query' in metadata and 'response' in metadata:
                        # Create a trace for the conversation
                        trace_id = langfuse_client.create_trace_id()
                        
                        # Create the event within the trace
                        langfuse_client.create_event(
                            name=event_name,
                            metadata=enhanced_metadata,
                            trace_id=trace_id
                        )
                        
                        # Try to create a generation if methods are available
                        if hasattr(langfuse_client, 'start_generation'):
                            try:
                                generation = langfuse_client.start_generation(
                                    name=f"{event_name}_generation",
                                    input=metadata.get('query', ''),
                                    metadata=enhanced_metadata
                                )
                                
                                # Update with output
                                if hasattr(langfuse_client, 'update_current_generation'):
                                    langfuse_client.update_current_generation(
                                        output=metadata.get('response', ''),
                                        metadata=enhanced_metadata
                                    )
                                
                                logger.debug(f"Langfuse generation created for: {event_name}")
                            except Exception as gen_error:
                                logger.debug(f"Generation creation failed: {gen_error}")
                    else:
                        # Simple event for non-chat events
                        langfuse_client.create_event(
                            name=event_name,
                            metadata=enhanced_metadata
                        )
                    
                    logger.debug(f"Langfuse event logged via create_event: {event_name}")
                    
                except Exception as e:
                    logger.warning(f"create_event failed: {e}")
                    # Fallback to basic logging
                    logger.info(f"Langfuse event (create_event failed): {event_name}")
                    logger.debug(f"Metadata: {metadata}")
            else:
                logger.warning(f"create_event method not available. Available methods: {available_methods}")
                # Fallback to basic logging
                logger.info(f"Langfuse event (method not available): {event_name}")
                logger.debug(f"Metadata: {metadata}")
            
        except Exception as e:
            logger.warning(f"Failed to log to Langfuse: {e}")
            # Fallback to basic logging
            logger.info(f"Langfuse event (fallback): {event_name}")
            logger.debug(f"Metadata: {metadata}")
    else:
        logger.debug(f"Langfuse client not available for event: {event_name}")

# Query preprocessing functions
def preprocess_query(query: str) -> str:
    """Preprocess query to improve retrieval accuracy."""
    # 1. Remove extra whitespace
    query = query.strip()
    
    # 2. Expand common abbreviations (customize based on your domain)
    abbreviations = {
        "AI": "artificial intelligence",
        "ML": "machine learning",
        "NLP": "natural language processing",
        "API": "application programming interface",
        "UI": "user interface",
        "UX": "user experience",
    }
    
    for abbr, full_form in abbreviations.items():
        query = query.replace(abbr, f"{abbr} {full_form}")
    
    # 3. Add context indicators for better retrieval
    if "?" in query:
        query = f"Question: {query}"
    elif any(word in query.lower() for word in ["how", "what", "when", "where", "why", "who"]):
        query = f"Information request: {query}"
    
    return query

def generate_query_variants(query: str) -> list:
    """Generate query variants for better retrieval."""
    variants = [query]
    
    # Add synonyms and related terms
    if "error" in query.lower():
        variants.append(query.replace("error", "problem"))
        variants.append(query.replace("error", "issue"))
    
    if "fix" in query.lower():
        variants.append(query.replace("fix", "solve"))
        variants.append(query.replace("fix", "resolve"))
    
    if "install" in query.lower():
        variants.append(query.replace("install", "setup"))
        variants.append(query.replace("install", "configure"))
    
    return variants

async def initialize_query_engine_for_ready_chatbot(chatbot_id: str):
    """
    Initializes a query engine by loading an existing knowledge graph index
    from its dedicated persistence directory and connecting to Neo4j.
    """
    logger.info(f"Attempting to re-initialize query engine for chatbot_id: {chatbot_id}")
    try:
        # *** IMPROVEMENT: Define the dedicated directory for this chatbot's metadata ***
        persist_dir = os.path.join(STORAGE_DIRECTORY, chatbot_id)
        if not os.path.exists(os.path.join(persist_dir, "index_store.json")):
             raise FileNotFoundError(f"Index metadata 'index_store.json' not found in {persist_dir}. The index may not have been built or persisted correctly.")

        # Run blocking Neo4jGraphStore initialization in a separate thread
        graph_store = await asyncio.to_thread(
            Neo4jGraphStore,
            username=NEO4J_USERNAME,
            password=NEO4J_PASSWORD,
            url=NEO4J_URI,
            database="neo4j",
        )
        
        # *** IMPROVEMENT: Create StorageContext with BOTH graph_store and persist_dir ***
        storage_context = StorageContext.from_defaults(
            graph_store=graph_store,
            persist_dir=persist_dir
        )
        
        logger.info(f"Loading index for chatbot {chatbot_id} from storage...")
        # Run blocking load_index_from_storage in a separate thread
        index = await asyncio.to_thread(
            load_index_from_storage,
            storage_context=storage_context,
        )

        if not isinstance(index, KnowledgeGraphIndex):
            raise TypeError(f"Loaded index for {chatbot_id} is not a KnowledgeGraphIndex, but {type(index)}.")

        logger.info(f"Successfully loaded KnowledgeGraphIndex with embeddings for chatbot {chatbot_id}.")

        query_engine = index.as_query_engine(
            response_synthesizer=get_response_synthesizer(
                text_qa_template=QA_TEMPLATE_WITH_CITATION,
            ),
            include_text=True,
            embedding_mode="hybrid",
            similarity_top_k=8,  # 増加：より多くの候補を取得
            retriever_mode="hybrid",  # ハイブリッド検索を明示的に指定
        )
        query_engines[chatbot_id] = query_engine
        logger.info(f"Query engine for {chatbot_id} re-initialized successfully.")

    except asyncio.CancelledError:
        logger.warning(f"Initialization of query engine for chatbot {chatbot_id} was cancelled.")
        raise # Re-raise to propagate the cancellation
    except Exception as e:
        logger.error(f"Failed to load index from storage for chatbot_id: {chatbot_id}. Error: {e}", exc_info=True)
        raise

def build_knowledge_graph(chatbot_id: str):
    """Builds the knowledge graph and persists its metadata."""
    chatbot = chatbots_db.get(chatbot_id)
    if not chatbot:
        logger.error(f"Chatbot {chatbot_id} not found for indexing.")
        return

    # Log indexing start to Langfuse
    log_to_langfuse(
        "build_knowledge_graph_start",
        {
            "chatbot_id": chatbot_id,
            "chatbot_name": chatbot.name if chatbot else "Unknown"
        },
        ["indexing", "knowledge_graph"]
    )

    try:
        logger.info(f"[Task Started] Building knowledge graph for chatbot: {chatbot_id}")
        chatbot.status = ChatbotStatus.INDEXING
        chatbot.current_step = IndexingStep.LOADING_DOCUMENTS
        save_chatbots_metadata()

        # Callback handler (No changes needed)
        # ...

        graph_store = Neo4jGraphStore(
            username=NEO4J_USERNAME,
            password=NEO4J_PASSWORD,
            url=NEO4J_URI,
            database="neo4j",
        )
        storage_context = StorageContext.from_defaults(graph_store=graph_store)

        input_dir = os.path.join(UPLOAD_DIRECTORY, chatbot_id)
        
        # Enhanced document loading with better error handling
        documents = []
        try:
            # Try to load all documents with explicit PDF reader if available
            file_extractor = {}
            if PDFReader is not None:
                pdf_reader = PDFReader()
                file_extractor[".pdf"] = pdf_reader
            
            reader = SimpleDirectoryReader(
                input_dir,
                file_extractor=file_extractor if file_extractor else None,
                recursive=True,  # Read subdirectories
                exclude_hidden=True,  # Skip hidden files
                required_exts=[".pdf", ".txt", ".md", ".docx"],  # Only process these file types
            )
            documents = reader.load_data()
            logger.info(f"Successfully loaded {len(documents)} documents")
            
        except Exception as e:
            logger.error(f"Error loading documents from {input_dir}: {e}")
            # Try to load documents one by one to identify problematic files
            for root, dirs, files in os.walk(input_dir):
                for file in files:
                    if file.lower().endswith(('.pdf', '.txt', '.md', '.docx')):
                        file_path = os.path.join(root, file)
                        try:
                            # Use appropriate reader based on file type
                            if file.lower().endswith('.pdf') and PDFReader is not None:
                                file_reader = SimpleDirectoryReader(
                                    input_files=[file_path],
                                    file_extractor={".pdf": PDFReader()}
                                )
                            else:
                                file_reader = SimpleDirectoryReader(input_files=[file_path])
                            
                            file_docs = file_reader.load_data()
                            if file_docs:  # Only add if documents were loaded successfully
                                documents.extend(file_docs)
                                logger.info(f"Successfully loaded: {file}")
                            else:
                                logger.warning(f"No content extracted from: {file}")
                        except Exception as file_error:
                            logger.warning(f"Failed to load {file}: {file_error}")
                            continue
        
        if not documents:
            raise ValueError(f"No documents could be loaded from {input_dir}")
        
        # Post-process documents to improve content quality
        processed_documents = []
        for doc in documents:
            try:
                # Basic validation and metadata enhancement
                if doc.text and len(doc.text.strip()) > 50:  # Minimum content length
                    # Ensure proper metadata without modifying the document
                    if not doc.metadata.get("file_name"):
                        doc.metadata["file_name"] = "unknown"
                    
                    processed_documents.append(doc)
                    logger.debug(f"Processed document: {doc.metadata.get('file_name', 'unknown')}")
                else:
                    logger.warning(f"Skipping document with insufficient content: {doc.metadata.get('file_name', 'unknown')}")
            except Exception as e:
                logger.warning(f"Error processing document {doc.metadata.get('file_name', 'unknown')}: {e}")
                # If processing fails, include the original document
                processed_documents.append(doc)
                continue
        
        documents = processed_documents
        logger.info(f"Processed {len(documents)} documents for indexing")

        logger.info(f"Chatbot {chatbot_id}: Building KnowledgeGraphIndex...")
        index = KnowledgeGraphIndex.from_documents(
            documents,
            storage_context=storage_context,
            max_triplets_per_chunk=4,  # 増加：より多くの関係性を抽出
            include_embeddings=True,
            show_progress=True,
            chunk_size=1024,  # チャンクサイズを明示的に設定
            chunk_overlap=100,  # オーバーラップを追加
        )
        
        # *** IMPROVEMENT: Persist the index metadata to its dedicated directory ***
        persist_dir = os.path.join(STORAGE_DIRECTORY, chatbot_id)
        logger.info(f"Persisting index metadata for chatbot {chatbot_id} to {persist_dir}")
        index.storage_context.persist(persist_dir=persist_dir)
        logger.info(f"Index metadata persisted successfully.")

        query_engine = index.as_query_engine(
            response_synthesizer=get_response_synthesizer(
                text_qa_template=QA_TEMPLATE_WITH_CITATION,
            ),
            include_text=True,
            embedding_mode="hybrid",
            similarity_top_k=8,  # 増加：より多くの候補を取得
            retriever_mode="hybrid",  # ハイブリッド検索を明示的に指定
        )
        
        query_engines[chatbot_id] = query_engine
        chatbot.status = ChatbotStatus.READY
        chatbot.current_step = None
        save_chatbots_metadata()
        logger.info(f"[Task Success] Knowledge graph for chatbot {chatbot_id} is ready.")

        # Log successful indexing to Langfuse
        log_to_langfuse(
            "build_knowledge_graph_success",
            {
                "chatbot_id": chatbot_id,
                "chatbot_name": chatbot.name,
                "status": "success",
                "documents_processed": len(documents) if 'documents' in locals() else 0
            },
            ["indexing", "knowledge_graph", "success"]
        )

    except Exception as e:
        logger.error(f"[Task Failed] Failed to build knowledge graph for {chatbot_id}: {e}", exc_info=True)
        
        # Log failed indexing to Langfuse
        log_to_langfuse(
            "build_knowledge_graph_failed",
            {
                "chatbot_id": chatbot_id,
                "chatbot_name": chatbot.name if chatbot else "Unknown",
                "status": "failed",
                "error": str(e),
                "error_type": type(e).__name__
            },
            ["indexing", "knowledge_graph", "error"]
        )
        
        if chatbot:
            chatbot.status = ChatbotStatus.FAILED
            chatbot.current_step = None
            chatbot.description = f"Indexing failed: {str(e)}" # Store error message
            save_chatbots_metadata()
    
    finally:
        # Ensure Langfuse data is flushed
        if langfuse_client:
            try:
                langfuse_client.flush()
                logger.debug("Langfuse data flushed successfully")
            except Exception as e:
                logger.warning(f"Failed to flush Langfuse data: {e}")

# --- Persistence Functions ---
async def load_chatbots_metadata():
    # ... (No changes here, but the logic inside that calls initialize_... is now more robust)
    global chatbots_db
    if not os.path.exists(CHATBOTS_METADATA_FILE):
        return
    try:
        with open(CHATBOTS_METADATA_FILE, "r", encoding="utf-8") as f: data = json.load(f)
        for bot_id, bot_data in data.items():
            if "status" in bot_data: bot_data["status"] = ChatbotStatus(bot_data["status"])
            if "current_step" in bot_data and bot_data["current_step"] is not None: bot_data["current_step"] = IndexingStep(bot_data["current_step"])
            chatbots_db[bot_id] = Chatbot(**bot_data)
        logger.info(f"Loaded {len(chatbots_db)} chatbots from {CHATBOTS_METADATA_FILE}")
        
        for bot_id, chatbot in chatbots_db.items():
            if chatbot.status == ChatbotStatus.READY:
                logger.info(f"Found READY chatbot: {bot_id}. Initializing its query engine.")
                try:
                    await initialize_query_engine_for_ready_chatbot(bot_id)
                except asyncio.CancelledError:
                    # If initialization is cancelled, mark as failed and re-raise
                    chatbot.status = ChatbotStatus.FAILED
                    logger.warning(f"Initialization for chatbot {bot_id} was cancelled. Marking as FAILED.")
                    # No need to re-raise here, as the outer loop will continue
                except Exception as e:
                    logger.error(f"Failed to initialize query engine for {bot_id} on startup. Marking as FAILED.")
                    chatbot.status = ChatbotStatus.FAILED
            elif chatbot.status == ChatbotStatus.INDEXING:
                 logger.warning(f"Chatbot {bot_id} was in INDEXING state during startup. Marking as FAILED.")
                 chatbot.status = ChatbotStatus.FAILED
        save_chatbots_metadata()
    except Exception as e:
        logger.error(f"A critical error occurred while loading chatbots metadata: {e}", exc_info=True)

def save_chatbots_metadata():
    with open(CHATBOTS_METADATA_FILE, "w", encoding="utf-8") as f:
        serializable_data = {bot_id: bot.dict() for bot_id, bot in chatbots_db.items()}
        json.dump(serializable_data, f, indent=4, ensure_ascii=False)

# --- FastAPI Events ---
@app.on_event("startup")
async def startup_event():
    # Download nltk data if not already present
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords')
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')

    await load_chatbots_metadata()

# --- API Endpoints ---
@app.get("/")
async def read_root():
    return {"message": "Welcome to the GraphLM Backend!"}

@app.get("/health")
async def health_check():
    """Health check endpoint that also reports Langfuse integration status."""
    langfuse_status = "enabled" if langfuse_client else "disabled"
    return {
        "status": "healthy",
        "langfuse_integration": langfuse_status,
        "version": "0.4.0"
    }

@app.post("/test-langfuse")
async def test_langfuse():
    """Test endpoint to verify Langfuse integration."""
    if not langfuse_client:
        return {"status": "error", "message": "Langfuse client not initialized"}
    
    try:
        # Get available methods for debugging
        available_methods = [method for method in dir(langfuse_client) if not method.startswith('_')]
        
        # Test logging to Langfuse
        log_to_langfuse(
            "test_event",
            {
                "test": True,
                "timestamp": "2024-01-01T00:00:00Z",
                "query": "test query",
                "response": "test response"
            },
            ["test", "api"]
        )
        
        # Force flush to ensure data is sent
        if hasattr(langfuse_client, 'flush'):
            langfuse_client.flush()
        
        return {
            "status": "success",
            "message": "Langfuse test event sent successfully",
            "available_methods": available_methods,
            "langfuse_version": getattr(langfuse_client, '__version__', 'unknown')
        }
    except Exception as e:
        return {
            "status": "error",
            "message": f"Langfuse test failed: {str(e)}",
            "available_methods": available_methods if 'available_methods' in locals() else []
        }

@app.get("/api/chatbots", response_model=List[Chatbot])
async def get_chatbots():
    return list(chatbots_db.values())

@app.put("/api/chatbots/{chatbot_id}", response_model=Chatbot)
async def update_chatbot(chatbot_id: str, chatbot_update: ChatbotUpdate):
    if chatbot_id not in chatbots_db:
        raise HTTPException(status_code=404, detail="Chatbot not found")
    
    existing_chatbot = chatbots_db[chatbot_id]
    if chatbot_update.name is not None:
        existing_chatbot.name = chatbot_update.name
    if chatbot_update.description is not None:
        existing_chatbot.description = chatbot_update.description
    
    save_chatbots_metadata()
    logger.info(f"Updated chatbot: {existing_chatbot.name} ({existing_chatbot.id})")
    return existing_chatbot

@app.delete("/api/chatbots/{chatbot_id}", status_code=204)
async def delete_chatbot(chatbot_id: str):
    if chatbot_id not in chatbots_db:
        raise HTTPException(status_code=404, detail="Chatbot not found")
    
    # Remove from in-memory store
    del chatbots_db[chatbot_id]
    if chatbot_id in query_engines:
        del query_engines[chatbot_id]

    # Remove associated files and storage
    chatbot_upload_dir = os.path.join(UPLOAD_DIRECTORY, chatbot_id)
    if os.path.exists(chatbot_upload_dir):
        shutil.rmtree(chatbot_upload_dir)
        logger.info(f"Removed upload directory: {chatbot_upload_dir}")

    chatbot_storage_dir = os.path.join(STORAGE_DIRECTORY, chatbot_id)
    if os.path.exists(chatbot_storage_dir):
        shutil.rmtree(chatbot_storage_dir)
        logger.info(f"Removed storage directory: {chatbot_storage_dir}")

    save_chatbots_metadata()
    logger.info(f"Deleted chatbot: {chatbot_id}")
    return

@app.post("/api/chatbots", response_model=Chatbot, status_code=201)
async def create_chatbot(name: str = Form(...), description: Optional[str] = Form(None)):
    new_chatbot_id = str(uuid.uuid4())
    new_chatbot = Chatbot(id=new_chatbot_id, name=name, description=description, status=ChatbotStatus.READY)
    chatbots_db[new_chatbot_id] = new_chatbot

    # *** IMPROVEMENT: Create dedicated storage directory for the new chatbot ***
    chatbot_storage_dir = os.path.join(STORAGE_DIRECTORY, new_chatbot_id)
    os.makedirs(chatbot_storage_dir, exist_ok=True)
    
    save_chatbots_metadata()
    logger.info(f"Created chatbot: {new_chatbot.name} ({new_chatbot.id})")
    return new_chatbot

# The rest of the API endpoints remain unchanged...
@app.post("/api/chatbots/{chatbot_id}/upload", status_code=202)
async def upload_knowledge_source(chatbot_id: str, background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    # Validate chatbot exists
    if chatbot_id not in chatbots_db: 
        raise HTTPException(status_code=404, detail="Chatbot not found")
    
    # Validate uploaded file
    validate_uploaded_file(file)
    
    # Create chatbot directory
    chatbot_dir = os.path.join(UPLOAD_DIRECTORY, chatbot_id)
    os.makedirs(chatbot_dir, exist_ok=True)
    
    # Sanitize filename and create safe file path
    safe_filename = sanitize_filename(file.filename or "uploaded_file")
    file_path = os.path.join(chatbot_dir, safe_filename)
    
    try:
        # Save file securely
        with open(file_path, "wb") as buffer: 
            shutil.copyfileobj(file.file, buffer)
        
        # Start background processing
        background_tasks.add_task(build_knowledge_graph, chatbot_id)
        
        logger.info(f"File uploaded successfully: {safe_filename} for chatbot {chatbot_id}")
        
    except Exception as e:
        logger.error(f"File upload failed for chatbot {chatbot_id}: {e}")
        # Clean up partial file if it exists
        if os.path.exists(file_path):
            os.remove(file_path)
        raise HTTPException(status_code=500, detail="File upload failed")
    finally:
        file.file.close()
    
    return {
        "message": f"File '{safe_filename}' processing started for chatbot '{chatbots_db[chatbot_id].name}'.",
        "filename": safe_filename
    }

@app.get("/api/chatbots/{chatbot_id}/indexing_progress", response_model=IndexingProgressResponse)
async def get_indexing_progress(chatbot_id: str):
    chatbot = chatbots_db.get(chatbot_id)
    if not chatbot: raise HTTPException(status_code=404, detail="Chatbot not found")
    return IndexingProgressResponse(total_nodes=chatbot.total_nodes or 0, processed_nodes=chatbot.processed_nodes or 0, status=chatbot.status, current_step=chatbot.current_step)

@app.post("/api/chat")
async def chat_with_bot(request: ChatRequest):
    chatbot = chatbots_db.get(request.chatbot_id)
    if not chatbot:
        raise HTTPException(status_code=404, detail="Chatbot not found")
    if chatbot.status != ChatbotStatus.READY:
        raise HTTPException(
            status_code=423,
            detail=f"Chatbot is not ready. Current status: {chatbot.status}",
        )
    query_engine = query_engines.get(request.chatbot_id)
    if not query_engine:
        raise HTTPException(
            status_code=404, detail="Query engine not found. Please check server logs."
        )

    # --- NEW STREAMING IMPLEMENTATION ---
    async def event_generator() -> AsyncGenerator[str, None]:
        # Initialize Langfuse trace
        trace = None
        generation = None
        
        try:
            logger.info(
                f"Received chat request for chatbot {request.chatbot_id} with query: {request.query}"
            )

            # Log chat query start to Langfuse
            log_to_langfuse(
                "chat_query_start",
                {
                    "chatbot_id": request.chatbot_id,
                    "chatbot_name": chatbots_db.get(request.chatbot_id, {}).name if request.chatbot_id in chatbots_db else "Unknown",
                    "query": request.query,
                    "status": "started"
                },
                ["chat", "graphrag", "start"]
            )

            # Step 1: Preprocess the query for better retrieval
            processed_query = preprocess_query(request.query)
            logger.info(f"Original query: {request.query}")
            logger.info(f"Processed query: {processed_query}")
            
            # Step 2: Get the full, non-streaming response first to ensure process completion
            response = await query_engine.aquery(processed_query)

            # Step 3: Extract all necessary data from the completed response
            logger.info(f"Response metadata: {response.metadata}")

            # Post-process response for better quality
            final_text = str(response)
            
            # Remove any duplicate sentences
            sentences = final_text.split('. ')
            seen_sentences = set()
            unique_sentences = []
            for sentence in sentences:
                sentence_clean = sentence.strip().lower()
                if sentence_clean not in seen_sentences and len(sentence_clean) > 10:
                    seen_sentences.add(sentence_clean)
                    unique_sentences.append(sentence.strip())
            
            final_text = '. '.join(unique_sentences)
            if not final_text.endswith('.'):
                final_text += '.'
            sources = []
            for node in response.source_nodes:
                file_name = node.metadata.get("file_name", "Unknown")
                url = f"/files/{request.chatbot_id}/{file_name}" if file_name != "Unknown" else None
                sources.append(
                    Source(
                        document_name=file_name,
                        page_number=node.metadata.get("page_label"),
                        snippet=node.get_content(metadata_mode="all"),
                        url=url,
                    ).dict()
                )
            
            formatted_graph_data = None
            raw_kg_rel_map = response.metadata.get("kg_rel_map") if response.metadata else None
            
            logger.info(f"Initial raw_kg_rel_map: {raw_kg_rel_map}")

            if not raw_kg_rel_map:
                for key, value in response.metadata.items():
                    if isinstance(value, dict) and "kg_rel_map" in value:
                        raw_kg_rel_map = value["kg_rel_map"]
                        logger.info(f"Found nested raw_kg_rel_map: {raw_kg_rel_map}")
                        break

            if raw_kg_rel_map:
                logger.info(f"Processing raw_kg_rel_map for graph: {raw_kg_rel_map}")
                # First, create a map of source_node_id to its text content
                source_text_map = {node.id_: node.get_content() for node in response.source_nodes}

                nodes_set = set()
                links = []
                
                # Known relationship types to exclude from links
                # RELATIONSHIP_TYPES_TO_EXCLUDE = {"INCLUDES", "FOCUSES_ON"} # Commented out for testing

                # The keys of raw_kg_rel_map are the source_node_ids
                for source_node_id, triplets in raw_kg_rel_map.items():
                    source_text = source_text_map.get(source_node_id, "")
                    for triplet in triplets:
                        # Expecting [relation, object]
                        if len(triplet) == 2:
                            rel, obj = triplet
                            subj = source_node_id # The subject is the key of the raw_kg_rel_map

                            nodes_set.add(subj)
                            nodes_set.add(obj) # Add object as a node

                            # Only add link if both source and target are valid (removed exclusion filter)
                            if subj and obj:
                                links.append({
                                    "source": subj,
                                    "target": obj,
                                    "label": rel,
                                    "source_text": source_text # Add the source text to the link
                                })
                        else:
                            # Log malformed triplets if necessary for debugging
                            logger.warning(f"Malformed triplet encountered: {triplet}")
                            continue # Skip malformed triplets

                # All collected nodes are considered entities for now.
                filtered_nodes_set = nodes_set 

                formatted_graph_data = {
                    "nodes": [{"id": node, "label": node} for node in filtered_nodes_set],
                    "links": links
                }

            # Log successful chat query to Langfuse
            log_to_langfuse(
                "chat_query_success",
                {
                    "chatbot_id": request.chatbot_id,
                    "query": request.query,
                    "response": final_text[:500] + "..." if len(final_text) > 500 else final_text,
                    "source_count": len(sources),
                    "graph_nodes": len(formatted_graph_data.get("nodes", [])) if formatted_graph_data else 0,
                    "graph_links": len(formatted_graph_data.get("links", [])) if formatted_graph_data else 0,
                    "status": "success"
                },
                ["chat", "graphrag", "success"]
            )

            # Step 3: Stream the extracted data piece by piece
            
            # Stream the final text char by char for a smoother effect
            for char in final_text:
                message_json = ChatResponse(
                    event=StreamEvent.MESSAGE, data={"text": char}
                ).dict()
                yield f"data: {json.dumps(message_json)}\n\n"

            # Stream sources
            sources_json = ChatResponse(event=StreamEvent.SOURCES, data={"sources": sources}).dict()
            yield f"data: {json.dumps(sources_json)}\n\n"

            # Stream graph data
            if formatted_graph_data:
                graph_json = ChatResponse(event=StreamEvent.GRAPH, data=formatted_graph_data).dict()
                yield f"data: {json.dumps(graph_json)}\n\n"

        except Exception as e:
            logger.error(
                f"Error during query for chatbot {request.chatbot_id}: {e}",
                exc_info=True,
            )
            
            # Log failed chat query to Langfuse
            log_to_langfuse(
                "chat_query_failed",
                {
                    "chatbot_id": request.chatbot_id,
                    "query": request.query,
                    "error": str(e),
                    "error_type": type(e).__name__,
                    "status": "failed"
                },
                ["chat", "graphrag", "error"]
            )
            
            error_data = {"error": "An error occurred while querying the chatbot."}
            error_json = ChatResponse(event=StreamEvent.DONE, data=error_data).dict()
            yield f"data: {json.dumps(error_json)}\n\n"
        finally:
            # Ensure Langfuse trace is finalized
            if langfuse_client:
                try:
                    langfuse_client.flush()
                    logger.debug("Langfuse data flushed successfully")
                except Exception as e:
                    logger.warning(f"Failed to flush Langfuse data: {e}")
            
            # Signal the end of the stream
            done_json = ChatResponse(event=StreamEvent.DONE, data={}).dict()
            yield f"data: {json.dumps(done_json)}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")

@app.get("/api/chatbots/{chatbot_id}/graph")
def get_graph_data(chatbot_id: str):
    """Fetches the entire knowledge graph data from Neo4j for a specific chatbot."""
    if chatbot_id not in chatbots_db:
        raise HTTPException(status_code=404, detail="Chatbot not found")

    logger.info(f"Fetching graph data for chatbot_id: {chatbot_id}")
    try:
        graph_store = Neo4jGraphStore(
            username=NEO4J_USERNAME,
            password=NEO4J_PASSWORD,
            url=NEO4J_URI,
            database="neo4j",
        )

        # More robust Cypher query to get exactly what we need for visualization
        query = """
        MATCH (n:Entity)-[r]->(m:Entity)
        RETURN 
            id(n) AS source_id, n.id AS source_label, 
            id(m) AS target_id, m.id AS target_label, 
            type(r) AS rel_type
        """
        
        result = graph_store.query(query)

        nodes = {}  # Use a dict to de-duplicate nodes
        edges = []
        if result:
            for record in result:
                source_id = record.get('source_id')
                source_label = record.get('source_label')
                target_id = record.get('target_id')
                target_label = record.get('target_label')
                rel_type = record.get('rel_type')

                if source_id and source_label:
                    nodes[source_id] = {"id": source_id, "label": source_label}
                if target_id and target_label:
                    nodes[target_id] = {"id": target_id, "label": target_label}
                
                if source_id and target_id and rel_type:
                    edges.append({
                        "source": source_id,
                        "target": target_id,
                        "label": rel_type
                    })

        nodes_list = list(nodes.values())

        return {"nodes": nodes_list, "links": edges} # Return "links" for react-force-graph

    except Exception as e:
        logger.error(f"Error fetching graph data for chatbot {chatbot_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="An error occurred while fetching graph data.")