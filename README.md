# GraphLM

GraphLM is a self-service GraphRAG chatbot building platform that allows employees to easily create and utilize highly accurate chatbots.

## Project Structure

- `backend/`: Contains the Python FastAPI backend application.
- `frontend/`: Contains the Next.js frontend application.

## Getting Started

### Backend Setup

1.  Navigate to the `backend` directory:
    ```bash
    cd backend
    ```
2.  Install dependencies using `pip` (or `poetry` if preferred):
    ```bash
    pip install -e .
    ```
3.  Run the FastAPI application:
    ```bash
    uvicorn main:app --reload
    ```
    The backend will be accessible at `http://localhost:8000`.

### Frontend Setup

1.  Navigate to the `frontend` directory:
    ```bash
    cd frontend
    ```
2.  Install dependencies using `npm`:
    ```bash
    npm install
    ```
3.  Run the Next.js development server:
    ```bash
    npm run dev
    ```
    The frontend will be accessible at `http://localhost:3000`.

## Features (Planned)

-   **Easy Chatbot Creation:** Intuitive UI for creating new chatbots.
-   **Local File & SharePoint Integration:** Seamlessly connect local documents and SharePoint content as knowledge sources.
-   **Notebook LM-style UI:** Chat interface with direct citation to source documents.
-   **Interactive Graph Visualization:** Explore knowledge graphs to understand relationships between information.

## Contributing

(Add contribution guidelines here)

## License

(Refer to LICENSE file)
