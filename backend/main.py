from fastapi import FastAPI, UploadFile, File, Form, HTTPException, BackgroundTasks
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
import shutil
from dotenv import load_dotenv
from typing import Dict, List, Optional, AsyncGenerator
import uuid
import logging
from enum import Enum
import json
import asyncio

from llama_index.core.chat_engine.types import StreamingAgentChatResponse

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



from openinference.instrumentation.llama_index import LlamaIndexInstrumentor

# Langfuse Integration (New)
LANGFUSE_PUBLIC_KEY = os.getenv("LANGFUSE_PUBLIC_KEY")
LANGFUSE_SECRET_KEY = os.getenv("LANGFUSE_SECRET_KEY")
LANGFUSE_HOST = os.getenv("LANGFUSE_HOST")

# if all([LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY, LANGFUSE_HOST]):
#     LlamaIndexInstrumentor().instrument()
#     logger.info("Langfuse OpenInference instrumentation initialized for LlamaIndex.")
# else:
#     logger.warning("Langfuse environment variables not fully set. Skipping Langfuse OpenInference integration.")
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
            response_mode="compact",
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
            response_mode="compact",
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
            chatbot.description = f"Indexing failed: {str(e)}" # Store error message
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
        try:
            logger.info(
                f"Received chat request for chatbot {request.chatbot_id} with query: {request.query}"
            )

            # Step 1: Get the full, non-streaming response first to ensure process completion
            response = await query_engine.aquery(request.query)

            # Step 2: Extract all necessary data from the completed response
            final_text = str(response)
            sources = [
                Source(
                    document_name=node.metadata.get("file_name", "Unknown"),
                    page_number=node.metadata.get("page_label"),
                    snippet=node.get_content(metadata_mode="all"),
                ).dict()
                for node in response.source_nodes
            ]
            
            # Extract graph data
            formatted_graph_data = None
            raw_kg_rel_map = response.metadata.get("kg_rel_map") if response.metadata else None
            if not raw_kg_rel_map:
                 for key, value in response.metadata.items():
                    if isinstance(value, dict) and "kg_rel_map" in value:
                        raw_kg_rel_map = value["kg_rel_map"]
                        break

            if raw_kg_rel_map:
                nodes_set = set()
                links = []
                for subject, triplets in raw_kg_rel_map.items():
                    nodes_set.add(subject)
                    for triplet in triplets:
                        if len(triplet) == 2:
                            relation, obj = triplet
                            nodes_set.add(obj)
                            links.append({"source": subject, "target": obj, "label": relation})
                        elif len(triplet) == 3:
                            s, relation, o = triplet
                            nodes_set.add(s)
                            nodes_set.add(o)
                            links.append({"source": s, "target": o, "label": relation})
                formatted_graph_data = {
                    "nodes": [{"id": node, "label": node} for node in nodes_set],
                    "links": links
                }

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
            error_data = {"error": "An error occurred while querying the chatbot."}
            error_json = ChatResponse(event=StreamEvent.DONE, data=error_data).dict()
            yield f"data: {json.dumps(error_json)}\n\n"
        finally:
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