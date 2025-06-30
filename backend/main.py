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
import json # Added for JSON persistence

# LlamaIndex imports
from llama_index.core import (
    Settings,
    Document,
    VectorStoreIndex,
    StorageContext,
    KnowledgeGraphIndex,
    SimpleDirectoryReader,
)
from llama_index.core.callbacks import CallbackManager, CBEventType, EventPayload
from llama_index.core.graph_stores import SimpleGraphStore
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.graph_stores.neo4j import Neo4jGraphStore

# Load environment variables
load_dotenv()

# --- Logging Setup ---
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)
logging.getLogger('llama_index').setLevel(logging.DEBUG) # Enable LlamaIndex debug logs

# --- Constants ---
UPLOAD_DIRECTORY = "./uploaded_files"
os.makedirs(UPLOAD_DIRECTORY, exist_ok=True)
CHATBOTS_METADATA_FILE = "./chatbots_metadata.json" # New constant for metadata file

# --- FastAPI App Initialization ---
app = FastAPI(
    title="GraphLM Backend",
    description="API for creating and managing GraphRAG-powered chatbots.",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Status Enums ---
class ChatbotStatus(str, Enum):
    INDEXING = "INDEXING"
    READY = "READY"
    FAILED = "FAILED"

class IndexingStep(str, Enum):
    LOADING_DOCUMENTS = "Loading Documents"
    PARSING_NODES = "Parsing Nodes"
    GENERATING_EMBEDDINGS = "Generating Embeddings"
    BUILDING_GRAPH = "Building Graph"

# --- API Models ---
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

class StatusResponse(BaseModel):
    status: ChatbotStatus

# --- In-Memory Stores (Placeholders) ---
chatbots_db: Dict[str, Chatbot] = {}
query_engines: Dict[str, any] = {}

# --- LlamaIndex Global Settings ---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")

if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY environment variable not set.")

Settings.llm = OpenAI(temperature=0, model="gpt-4.1-nano", api_key=OPENAI_API_KEY)
Settings.embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
Settings.num_workers = 4 # Set number of workers for parallel processing

# --- Helper Functions (defined before Persistence Functions for proper scope) ---
def initialize_query_engine_for_ready_chatbot(chatbot_id: str):
    """Initializes query engine for a chatbot that is already READY (data exists in Neo4j)."""
    graph_store = Neo4jGraphStore(
        username=NEO4J_USERNAME,
        password=NEO4J_PASSWORD,
        url=NEO4J_URI,
        database="neo4j",
    )
    storage_context = StorageContext.from_defaults(graph_store=graph_store)

    # Load index from existing graph data
    index = KnowledgeGraphIndex.from_documents(
        documents=[], # Pass an empty list of documents as we are loading from existing graph
        storage_context=storage_context,
    )

    query_engine = index.as_query_engine(
        include_text=True,
        response_mode="tree_summarize",
        embedding_mode="hybrid",
        similarity_top_k=5,
    )
    query_engines[chatbot_id] = query_engine
    logger.info(f"Query engine for {chatbot_id} re-initialized from existing graph.")

def build_knowledge_graph(chatbot_id: str):
    """Builds the knowledge graph for a chatbot in the background."""
    chatbot = chatbots_db.get(chatbot_id)
    if not chatbot:
        logger.error(f"Chatbot {chatbot_id} not found for indexing.")
        return

    try:
        logger.info(f"[Task Started] Building knowledge graph for chatbot: {chatbot_id}")
        chatbot.status = ChatbotStatus.INDEXING
        chatbot.total_nodes = 0
        chatbot.processed_nodes = 0
        chatbot.current_step = IndexingStep.LOADING_DOCUMENTS
        save_chatbots_metadata() # Save status change
        logger.info(f"Chatbot {chatbot_id} status set to INDEXING. Initial progress: {chatbot.processed_nodes}/{chatbot.total_nodes}")

        # Custom callback handler for progress
        class IndexingProgressCallback(CallbackManager):
            def __init__(self, chatbot_id: str):
                super().__init__()
                self.chatbot_id = chatbot_id
                self.on_event_end = self._on_event_end

            def _on_event_end(self, event_type: CBEventType, payload: Optional[Dict] = None, **kwargs) -> None:
                chatbot = chatbots_db.get(self.chatbot_id)
                if not chatbot:
                    logger.warning(f"Chatbot {self.chatbot_id} not found in callback.")
                    return

                if event_type == CBEventType.NODE_PARSING:
                    nodes = payload.get(EventPayload.NODES)
                    if nodes:
                        chatbot.total_nodes = len(nodes)
                        chatbot.current_step = IndexingStep.PARSING_NODES
                        logger.info(f"Chatbot {self.chatbot_id}: Total nodes to process: {chatbot.total_nodes}. Current step: {chatbot.current_step}")
                    else:
                        logger.warning(f"Chatbot {self.chatbot_id}: NODE_PARSING event with no nodes in payload.")
                elif event_type == CBEventType.EMBEDDING:
                    chatbot.processed_nodes += 1
                    chatbot.current_step = IndexingStep.GENERATING_EMBEDDINGS
                    logger.info(f"Chatbot {self.chatbot_id}: Processed nodes: {chatbot.processed_nodes}/{chatbot.total_nodes}. Current step: {chatbot.current_step}")
                elif event_type == CBEventType.RETRIEVE:
                    chatbot.current_step = IndexingStep.BUILDING_GRAPH
                    logger.info(f"Chatbot {self.chatbot_id}: Current step: {chatbot.current_step}")
                else:
                    logger.debug(f"Chatbot {self.chatbot_id}: Unhandled event type: {event_type}")
                save_chatbots_metadata() # Save progress periodically


        callback_manager = IndexingProgressCallback(chatbot_id)
        
        graph_store = Neo4jGraphStore(
            username=NEO4J_USERNAME,
            password=NEO4J_PASSWORD,
            url=NEO4J_URI,
            database="neo4j",
        )
        storage_context = StorageContext.from_defaults(graph_store=graph_store)

        input_dir = os.path.join(UPLOAD_DIRECTORY, chatbot_id)
        if not os.path.exists(input_dir) or not os.listdir(input_dir):
            logger.warning(f"No documents found for chatbot {chatbot_id}")
            raise ValueError("No documents to index.")

        logger.info(f"Chatbot {chatbot_id}: Attempting to load documents from {input_dir}")
        documents = SimpleDirectoryReader(input_dir).load_data()
        if not documents:
            logger.warning(f"Chatbot {chatbot_id}: Could not load any documents from {input_dir}")
            raise ValueError("Could not load any documents.")
        logger.info(f"Chatbot {chatbot_id}: Successfully loaded {len(documents)} document(s). Proceeding to build index.")

        logger.info(f"Chatbot {chatbot_id}: Building KnowledgeGraphIndex from {len(documents)} documents...")
        index = KnowledgeGraphIndex.from_documents(
            documents,
            storage_context=storage_context,
            max_triplets_per_chunk=2,
            include_embeddings=True,
            callback_manager=callback_manager, # Pass the callback manager to the index
        )
        logger.info(f"Chatbot {chatbot_id}: KnowledgeGraphIndex built successfully.")

        query_engine = index.as_query_engine(
            include_text=True,
            response_mode="tree_summarize",
            embedding_mode="hybrid",
            similarity_top_k=5,
        )
        
        query_engines[chatbot_id] = query_engine
        chatbot.status = ChatbotStatus.READY
        chatbot.current_step = None # Reset step on completion
        save_chatbots_metadata() # Save final status
        logger.info(f"[Task Success] Knowledge graph for chatbot {chatbot_id} is ready.")

    except Exception as e:
        logger.error(f"[Task Failed] Failed to build knowledge graph for {chatbot_id}: {e}")
        chatbot.status = ChatbotStatus.FAILED
        chatbot.current_step = None # Reset step on failure
        save_chatbots_metadata() # Save failed status

# --- Persistence Functions (defined after Helper Functions for proper scope) ---
def load_chatbots_metadata():
    global chatbots_db
    if os.path.exists(CHATBOTS_METADATA_FILE):
        with open(CHATBOTS_METADATA_FILE, "r", encoding="utf-8") as f:
            try:
                data = json.load(f)
                for bot_id, bot_data in data.items():
                    # Ensure enums are correctly loaded
                    if "status" in bot_data: 
                        bot_data["status"] = ChatbotStatus(bot_data["status"])
                    if "current_step" in bot_data and bot_data["current_step"] is not None:
                        bot_data["current_step"] = IndexingStep(bot_data["current_step"])
                    chatbots_db[bot_id] = Chatbot(**bot_data)
                logger.info(f"Loaded {len(chatbots_db)} chatbots from {CHATBOTS_METADATA_FILE}")
                
                # Initialize query engines for READY chatbots on startup
                # Initialize query engines for READY chatbots on startup
                for bot_id, chatbot in chatbots_db.items():
                    # Only attempt to re-initialize if it's READY or FAILED (to try recovery)
                    if chatbot.status == ChatbotStatus.READY or chatbot.status == ChatbotStatus.FAILED:
                        logger.info(f"Attempting to initialize query engine for chatbot: {bot_id} (current status: {chatbot.status})")
                        try:
                            initialize_query_engine_for_ready_chatbot(bot_id) # Call the helper function
                            logger.info(f"Query engine initialized for {bot_id}")
                            # If successful, ensure status is READY
                            if chatbot.status == ChatbotStatus.FAILED: # If it was failed, and now succeeded
                                chatbot.status = ChatbotStatus.READY
                                logger.info(f"Chatbot {bot_id} status changed from FAILED to READY after successful re-initialization.")
                        except Exception as e:
                            logger.error(f"Failed to initialize query engine for {bot_id}: {e}")
                            chatbot.status = ChatbotStatus.FAILED # Mark as failed if re-initialization fails
                else:
                        logger.info(f"Skipping query engine initialization for chatbot {bot_id} (status: {chatbot.status})")
                save_chatbots_metadata() # Save any status changes made during re-initialization

            except json.JSONDecodeError as e:
                logger.error(f"Error decoding JSON from {CHATBOTS_METADATA_FILE}: {e}")
            except Exception as e:
                logger.error(f"Error loading chatbots metadata: {e}")
    else:
        logger.info(f"No existing chatbot metadata file found at {CHATBOTS_METADATA_FILE}")

def save_chatbots_metadata():
    with open(CHATBOTS_METADATA_FILE, "w", encoding="utf-8") as f:
        # Convert Chatbot objects to dictionaries for JSON serialization
        serializable_data = {bot_id: bot.dict() for bot_id, bot in chatbots_db.items()}
        json.dump(serializable_data, f, indent=4, ensure_ascii=False)
    logger.info(f"Saved {len(chatbots_db)} chatbots to {CHATBOTS_METADATA_FILE}")

# Load chatbots metadata on startup
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
    new_chatbot = Chatbot(id=new_chatbot_id, name=name, description=description, status=ChatbotStatus.READY, total_nodes=0, processed_nodes=0, current_step=None)
    chatbots_db[new_chatbot_id] = new_chatbot
    save_chatbots_metadata() # Save new chatbot
    logger.info(f"Created chatbot: {new_chatbot}")
    return new_chatbot

@app.post("/api/chatbots/{chatbot_id}/upload", status_code=202) # 202 Accepted
async def upload_knowledge_source(chatbot_id: str, background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    if chatbot_id not in chatbots_db:
        raise HTTPException(status_code=404, detail="Chatbot not found")
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file uploaded")

    chatbot_dir = os.path.join(UPLOAD_DIRECTORY, chatbot_id)
    os.makedirs(chatbot_dir, exist_ok=True)
    file_path = os.path.join(chatbot_dir, file.filename)

    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        logger.info(f"File '{file.filename}' uploaded to '{file_path}'")
        
        # Start the knowledge graph building in the background
        background_tasks.add_task(build_knowledge_graph, chatbot_id)

    finally:
        file.file.close()

    return {"message": f"File processing started for chatbot '{chatbot_id}'."}

class IndexingProgressResponse(BaseModel):
    total_nodes: int
    processed_nodes: int
    status: ChatbotStatus
    current_step: Optional[IndexingStep] = None

@app.get("/api/chatbots/{chatbot_id}/indexing_progress", response_model=IndexingProgressResponse)
async def get_indexing_progress(chatbot_id: str):
    chatbot = chatbots_db.get(chatbot_id)
    if not chatbot:
        raise HTTPException(status_code=404, detail="Chatbot not found")
    return IndexingProgressResponse(
        total_nodes=chatbot.total_nodes,
        processed_nodes=chatbot.processed_nodes,
        status=chatbot.status,
        current_step=chatbot.current_step
    )

@app.post("/api/chat", response_model=ChatResponse)
async def chat_with_bot(request: ChatRequest):
    logger.info(f"Received chat request for chatbot: {request.chatbot_id}")
    chatbot = chatbots_db.get(request.chatbot_id)
    if not chatbot:
        raise HTTPException(status_code=404, detail="Chatbot not found")

    if chatbot.status == ChatbotStatus.INDEXING:
        raise HTTPException(status_code=423, detail="Chatbot is currently processing documents. Please try again later.")
    if chatbot.status == ChatbotStatus.FAILED:
        raise HTTPException(status_code=500, detail="Chatbot processing failed. Please try uploading the document again.")

    query_engine = query_engines.get(request.chatbot_id)
    if query_engine is None:
        raise HTTPException(status_code=404, detail="Query engine not found. Please upload a document first.")

    logger.info(f"Querying with: '{request.query}'")
    response = await query_engine.aquery(request.query)

    sources = [
        Source(
            document_name=node.metadata.get('file_name', 'Unknown'),
            page_number=node.metadata.get('page_label'),
            snippet=node.get_content(metadata_mode="llm"),
        )
        for node in response.source_nodes
    ]

    return ChatResponse(response=str(response), sources=sources)