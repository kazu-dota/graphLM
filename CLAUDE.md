# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Setup

### Backend (FastAPI + Python)
- **Install dependencies**: `pip install -e .` (from backend directory)
- **Run development server**: `uvicorn main:app --reload` (from backend directory)
- **Virtual environment**: Use `.venv\Scripts\Activate.ps1` to activate virtual environment
- **Configuration**: Requires environment variables for OpenAI API, Neo4j database, and optionally Langfuse

### Frontend (Next.js + TypeScript)
- **Install dependencies**: `npm install` (from frontend directory)
- **Run development server**: `npm run dev --turbopack` (enables Turbopack for faster builds)
- **Build for production**: `npm run build`
- **Start production server**: `npm start`
- **Linting**: `npm run lint`

## Architecture Overview

### Backend Architecture
- **FastAPI application** in `backend/main.py` - single-file architecture with all endpoints
- **GraphRAG system** using LlamaIndex and Neo4j for knowledge graph storage
- **Streaming chat responses** with Server-Sent Events (SSE) for real-time communication
- **File upload handling** with background processing for document indexing
- **Persistence layer** using JSON metadata files and Neo4j graph database

### Frontend Architecture
- **Next.js with TypeScript** using Pages Router (not App Router)
- **Material-UI (MUI)** for UI components and styling
- **Component-based architecture** with clear separation of concerns:
  - `ChatInterface.tsx` - Handles streaming chat and message display
  - `GraphPanel.tsx` - Manages graph visualization tabs
  - `GraphView.tsx` - Renders interactive force-directed graphs
  - `ChatbotList.tsx` - Manages chatbot creation and listing
  - `KnowledgeSourceUpload.tsx` - Handles file uploads
- **API layer** in `utils/api.ts` for backend communication

### Key Technical Patterns
- **Streaming responses**: Backend uses FastAPI's `StreamingResponse` with SSE format
- **Knowledge graph integration**: Uses Neo4j for graph storage and LlamaIndex for RAG
- **File processing**: Background tasks handle document indexing asynchronously
- **State management**: React hooks with local state, no external state management library
- **Graph visualization**: Uses `react-force-graph-2d` for interactive knowledge graphs

## Common Development Commands

### Running the Full Application
1. **Start backend**: `cd backend && uvicorn main:app --reload`
2. **Start frontend**: `cd frontend && npm run dev`
3. **Access application**: Frontend at `http://localhost:3000`, Backend at `http://localhost:8000`

### Testing and Quality Assurance
- **Frontend linting**: `npm run lint` (from frontend directory)
- **Backend testing**: No specific test command configured - check if pytest is available
- **Type checking**: Frontend uses TypeScript with strict mode enabled

## Database and External Services

### Neo4j Database
- Required for knowledge graph storage
- Configure via environment variables: `NEO4J_URI`, `NEO4J_USERNAME`, `NEO4J_PASSWORD`
- Used for entity relationships and graph queries

### OpenAI Integration
- Required for LLM functionality
- Configure via `OPENAI_API_KEY` environment variable
- Currently using GPT-4 model for chat responses

### Langfuse (Optional)
- LLM observability and monitoring
- Configure via `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY`, `LANGFUSE_HOST`
- Automatically traces chat queries, knowledge graph building, and system performance
- Check integration status via `/health` endpoint
- Test integration via `POST /test-langfuse` endpoint
- Traces include: chat queries (start/success/failed), knowledge graph building (start/success/failed)
- Each trace includes metadata: query, response, source count, graph data, error details

## File Storage Structure
- **Uploaded files**: `backend/uploaded_files/{chatbot_id}/`
- **Index storage**: `backend/storage/{chatbot_id}/`
- **Metadata**: `backend/chatbots_metadata.json`

## Development Workflow
1. **Creating new chatbots**: Use the frontend form to create, then upload knowledge sources
2. **Adding new endpoints**: Follow the existing FastAPI patterns in `main.py`
3. **Frontend components**: Use Material-UI components and follow existing TypeScript patterns
4. **Graph visualization**: Extend `GraphView.tsx` for new graph features
5. **Chat functionality**: Streaming is handled via SSE - maintain the existing event structure

## RAG Optimization Features
- **Enhanced search parameters**: `similarity_top_k=8`, hybrid retrieval mode
- **Improved knowledge graph**: `max_triplets_per_chunk=4`, optimized chunk size (1024) and overlap (100)
- **Query preprocessing**: Automatic query expansion, abbreviation handling, context indicators
- **Response post-processing**: Duplicate sentence removal, quality improvements
- **Better document parsing**: Enhanced file type support, content cleaning
- **Comprehensive prompts**: Detailed instructions for better LLM responses

## Important Notes
- **Single-file backend**: All backend logic is in `main.py` - consider refactoring for larger features
- **No authentication**: Current implementation has no user authentication system
- **CORS enabled**: Backend allows all origins for development
- **Graph processing**: Knowledge graphs are built asynchronously after file upload
- **Streaming protocol**: Chat responses use specific event types: `message`, `sources`, `graph`, `done`
- **RAG optimizations**: Multiple layers of preprocessing and post-processing for improved accuracy