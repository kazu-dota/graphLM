from fastapi import FastAPI, UploadFile, File, Form, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
import shutil
from dotenv import load_dotenv
from typing import Dict, List, Optional
import uuid
import logging
from enum import Enum
import json

# LlamaIndex imports
from llama_index.core import (
    Settings,
    StorageContext,
    KnowledgeGraphIndex,
    SimpleDirectoryReader,
    load_index_from_storage,
)
from llama_index.core.callbacks import CallbackManager, CBEventType, EventPayload
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.graph_stores.neo4j import Neo4jGraphStore

# Load environment variables
load_dotenv()

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("main") # Use a named logger
# logging.getLogger('llama_index.core').setLevel(logging.DEBUG)

# --- Constants ---
UPLOAD_DIRECTORY = "./uploaded_files"
STORAGE_DIRECTORY = "./storage" # *** NEW: Directory to store index metadata ***
os.makedirs(UPLOAD_DIRECTORY, exist_ok=True)
os.makedirs(STORAGE_DIRECTORY, exist_ok=True)
CHATBOTS_METADATA_FILE = "./chatbots_metadata.json"

# --- FastAPI App Initialization ---
app = FastAPI(
    title="GraphLM Backend",
    description="API for creating and managing GraphRAG-powered chatbots.",
    version="0.4.0", # Version bump to reflect improvements
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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

class ChatRequest(BaseModel):
    chatbot_id: str
    query: str

class Source(BaseModel):
    document_name: str
    page_number: Optional[int] = None
    snippet: Optional[str] = None

class ChatResponse(BaseModel):
    response: str
    sources: List[Source]

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

Settings.llm = OpenAI(temperature=0, model="gpt-4-turbo", api_key=OPENAI_API_KEY)
Settings.embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
Settings.num_workers = 4

# --- Helper Functions ---
def initialize_query_engine_for_ready_chatbot(chatbot_id: str):
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

        graph_store = Neo4jGraphStore(
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
        index = load_index_from_storage(
            storage_context=storage_context,
        )

        if not isinstance(index, KnowledgeGraphIndex):
            raise TypeError(f"Loaded index for {chatbot_id} is not a KnowledgeGraphIndex, but {type(index)}.")

        logger.info(f"Successfully loaded KnowledgeGraphIndex with embeddings for chatbot {chatbot_id}.")

        query_engine = index.as_query_engine(
            include_text=True,
            response_mode="tree_summarize",
            embedding_mode="hybrid",
            similarity_top_k=5,
        )
        query_engines[chatbot_id] = query_engine
        logger.info(f"Query engine for {chatbot_id} re-initialized successfully.")

    except Exception as e:
        logger.error(f"Failed to load index from storage for chatbot_id: {chatbot_id}. Error: {e}", exc_info=True)
        raise

def build_knowledge_graph(chatbot_id: str):
    """Builds the knowledge graph and persists its metadata."""
    chatbot = chatbots_db.get(chatbot_id)
    if not chatbot:
        logger.error(f"Chatbot {chatbot_id} not found for indexing.")
        return

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
        # ... (document loading logic is the same)
        documents = SimpleDirectoryReader(input_dir).load_data()

        logger.info(f"Chatbot {chatbot_id}: Building KnowledgeGraphIndex...")
        index = KnowledgeGraphIndex.from_documents(
            documents,
            storage_context=storage_context,
            max_triplets_per_chunk=2,
            include_embeddings=True,
            show_progress=True,
        )
        
        # *** IMPROVEMENT: Persist the index metadata to its dedicated directory ***
        persist_dir = os.path.join(STORAGE_DIRECTORY, chatbot_id)
        logger.info(f"Persisting index metadata for chatbot {chatbot_id} to {persist_dir}")
        index.storage_context.persist(persist_dir=persist_dir)
        logger.info(f"Index metadata persisted successfully.")

        query_engine = index.as_query_engine(
            include_text=True,
            response_mode="tree_summarize",
            embedding_mode="hybrid",
            similarity_top_k=5,
        )
        
        query_engines[chatbot_id] = query_engine
        chatbot.status = ChatbotStatus.READY
        chatbot.current_step = None
        save_chatbots_metadata()
        logger.info(f"[Task Success] Knowledge graph for chatbot {chatbot_id} is ready.")

    except Exception as e:
        logger.error(f"[Task Failed] Failed to build knowledge graph for {chatbot_id}: {e}", exc_info=True)
        if chatbot:
            chatbot.status = ChatbotStatus.FAILED
            chatbot.current_step = None
            save_chatbots_metadata()

# --- Persistence Functions ---
def load_chatbots_metadata():
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
                    initialize_query_engine_for_ready_chatbot(bot_id)
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
    load_chatbots_metadata()

# --- API Endpoints ---
@app.get("/")
async def read_root():
    return {"message": "Welcome to the GraphLM Backend!"}

@app.get("/api/chatbots", response_model=List[Chatbot])
async def get_chatbots():
    return list(chatbots_db.values())

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
    if chatbot_id not in chatbots_db: raise HTTPException(status_code=404, detail="Chatbot not found")
    chatbot_dir = os.path.join(UPLOAD_DIRECTORY, chatbot_id)
    os.makedirs(chatbot_dir, exist_ok=True)
    file_path = os.path.join(chatbot_dir, file.filename)
    try:
        with open(file_path, "wb") as buffer: shutil.copyfileobj(file.file, buffer)
        background_tasks.add_task(build_knowledge_graph, chatbot_id)
    finally:
        file.file.close()
    return {"message": f"File processing started for chatbot '{chatbots_db[chatbot_id].name}'."}

@app.get("/api/chatbots/{chatbot_id}/indexing_progress", response_model=IndexingProgressResponse)
async def get_indexing_progress(chatbot_id: str):
    chatbot = chatbots_db.get(chatbot_id)
    if not chatbot: raise HTTPException(status_code=404, detail="Chatbot not found")
    return IndexingProgressResponse(total_nodes=chatbot.total_nodes or 0, processed_nodes=chatbot.processed_nodes or 0, status=chatbot.status, current_step=chatbot.current_step)

@app.post("/api/chat", response_model=ChatResponse)
async def chat_with_bot(request: ChatRequest):
    chatbot = chatbots_db.get(request.chatbot_id)
    if not chatbot: raise HTTPException(status_code=404, detail="Chatbot not found")
    if chatbot.status != ChatbotStatus.READY: raise HTTPException(status_code=423, detail=f"Chatbot is not ready. Current status: {chatbot.status}")
    query_engine = query_engines.get(request.chatbot_id)
    if not query_engine: raise HTTPException(status_code=404, detail="Query engine not found. Please check server logs.")
    try:
        response = await query_engine.aquery(request.query)
        sources = [Source(document_name=node.metadata.get('file_name', 'Unknown'), page_number=node.metadata.get('page_label'), snippet=node.get_content(metadata_mode="all")) for node in response.source_nodes]
        return ChatResponse(response=str(response), sources=sources)
    except Exception as e:
        logger.error(f"Error during query for chatbot {request.chatbot_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="An error occurred while querying the chatbot.")