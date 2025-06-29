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

# LlamaIndex imports
from llama_index.core import (
    Settings,
    Document,
    VectorStoreIndex,
    StorageContext,
    KnowledgeGraphIndex,
    SimpleDirectoryReader,
)
from llama_index.core.graph_stores import SimpleGraphStore
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.graph_stores.neo4j import Neo4jGraphStore

# Load environment variables
load_dotenv()

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Constants ---
UPLOAD_DIRECTORY = "./uploaded_files"
os.makedirs(UPLOAD_DIRECTORY, exist_ok=True)

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

# --- Status Enum ---
class ChatbotStatus(str, Enum):
    INDEXING = "INDEXING"
    READY = "READY"
    FAILED = "FAILED"

# --- API Models ---
class Chatbot(BaseModel):
    id: str
    name: str
    description: Optional[str] = None
    status: ChatbotStatus = ChatbotStatus.READY

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
chatbot_statuses: Dict[str, ChatbotStatus] = {}
query_engines: Dict[str, any] = {}

# --- LlamaIndex Global Settings ---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")

if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY environment variable not set.")

Settings.llm = OpenAI(temperature=0, model="gpt-4.1-nano", api_key=OPENAI_API_KEY)
Settings.embed_model = HuggingFaceEmbedding(model_name="pkshatech/GLuCoSE-base-ja")

# --- Helper Functions ---
def build_knowledge_graph(chatbot_id: str):
    """Builds the knowledge graph for a chatbot in the background."""
    try:
        logger.info(f"[Task Started] Building knowledge graph for chatbot: {chatbot_id}")
        chatbot_statuses[chatbot_id] = ChatbotStatus.INDEXING
        
        # This function is now synchronous and will block the background worker
        query_engine = get_query_engine_for_chatbot(chatbot_id)
        
        if query_engine:
            query_engines[chatbot_id] = query_engine
            chatbot_statuses[chatbot_id] = ChatbotStatus.READY
            logger.info(f"[Task Success] Knowledge graph for chatbot {chatbot_id} is ready.")
        else:
            raise ValueError("Query engine could not be created.")

    except Exception as e:
        logger.error(f"[Task Failed] Failed to build knowledge graph for {chatbot_id}: {e}")
        chatbot_statuses[chatbot_id] = ChatbotStatus.FAILED

def get_query_engine_for_chatbot(chatbot_id: str):
    """Initializes and returns a query engine for a given chatbot ID."""
    logger.info(f"Creating query engine for chatbot: {chatbot_id}")
    
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
        return None

    documents = SimpleDirectoryReader(input_dir).load_data()
    if not documents:
        logger.warning(f"Could not load any documents from {input_dir}")
        return None

    logger.info(f"Loaded {len(documents)} document(s). Building index...")

    index = KnowledgeGraphIndex.from_documents(
        documents,
        storage_context=storage_context,
        max_triplets_per_chunk=2,
        include_embeddings=True,
    )

    query_engine = index.as_query_engine(
        include_text=True,
        response_mode="tree_summarize",
        embedding_mode="hybrid",
        similarity_top_k=5,
    )
    return query_engine

# --- API Endpoints ---
@app.get("/")
async def read_root():
    return {"message": "Welcome to the GraphLM Backend!"}

@app.get("/api/chatbots", response_model=List[Chatbot])
async def get_chatbots():
    bots_with_status = []
    for bot_id, bot in chatbots_db.items():
        bot.status = chatbot_statuses.get(bot_id, ChatbotStatus.READY)
        bots_with_status.append(bot)
    return bots_with_status

@app.post("/api/chatbots", response_model=Chatbot, status_code=201)
async def create_chatbot(name: str = Form(...), description: Optional[str] = Form(None)):
    new_chatbot_id = str(uuid.uuid4())
    new_chatbot = Chatbot(id=new_chatbot_id, name=name, description=description)
    chatbots_db[new_chatbot_id] = new_chatbot
    chatbot_statuses[new_chatbot_id] = ChatbotStatus.READY
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

@app.get("/api/chatbots/{chatbot_id}/status", response_model=StatusResponse)
async def get_chatbot_status(chatbot_id: str):
    if chatbot_id not in chatbots_db:
        raise HTTPException(status_code=404, detail="Chatbot not found")
    status = chatbot_statuses.get(chatbot_id, ChatbotStatus.READY)
    return StatusResponse(status=status)

@app.post("/api/chat", response_model=ChatResponse)
async def chat_with_bot(request: ChatRequest):
    logger.info(f"Received chat request for chatbot: {request.chatbot_id}")
    if request.chatbot_id not in chatbots_db:
        raise HTTPException(status_code=404, detail="Chatbot not found")

    status = chatbot_statuses.get(request.chatbot_id)
    if status == ChatbotStatus.INDEXING:
        raise HTTPException(status_code=423, detail="Chatbot is currently processing documents. Please try again later.")
    if status == ChatbotStatus.FAILED:
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