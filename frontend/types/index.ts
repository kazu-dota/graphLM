// Common types for the GraphRAG application

export interface Chatbot {
  id: string;
  name: string;
  description?: string;
  status: ChatbotStatus;
  total_nodes?: number;
  processed_nodes?: number;
  current_step?: IndexingStep;
  created_at?: string;
  updated_at?: string;
}

export enum ChatbotStatus {
  INDEXING = "INDEXING",
  READY = "READY",
  FAILED = "FAILED"
}

export enum IndexingStep {
  LOADING_DOCUMENTS = "Loading Documents",
  PARSING_NODES = "Parsing Nodes",
  GENERATING_EMBEDDINGS = "Generating Embeddings",
  BUILDING_GRAPH = "Building Graph"
}

export interface Message {
  id: string;
  sender: 'user' | 'bot';
  content: string;
  timestamp: number;
  status?: 'pending' | 'completed' | 'error';
  sources?: Source[];
  graph_data?: GraphData;
  error?: string;
}

export interface Source {
  id: string;
  filename: string;
  page_number?: number;
  snippet: string;
  url?: string;
  relevance_score?: number;
}

export interface GraphData {
  nodes: GraphNode[];
  links: GraphLink[];
}

export interface GraphNode {
  id: string;
  label: string;
  type?: string;
  size?: number;
  color?: string;
  x?: number;
  y?: number;
  [key: string]: any;
}

export interface GraphLink {
  source: string;
  target: string;
  label?: string;
  weight?: number;
  color?: string;
  [key: string]: any;
}

export interface ChatRequest {
  chatbot_id: string;
  query: string;
  conversation_id?: string;
}

export interface ChatResponse {
  event: StreamEvent;
  data: StreamData;
}

export enum StreamEvent {
  MESSAGE = "message",
  SOURCES = "sources", 
  GRAPH = "graph",
  DONE = "done",
  ERROR = "error"
}

export interface StreamData {
  text?: string;
  sources?: Source[];
  graph?: GraphData;
  error?: string;
  finished?: boolean;
}

export interface IndexingProgress {
  total_nodes: number;
  processed_nodes: number;
  status: ChatbotStatus;
  current_step?: IndexingStep;
  progress_percentage?: number;
}

export interface ApiError {
  error: string;
  message: string;
  status_code?: number;
  timestamp?: number;
  details?: any;
}

export interface UploadResponse {
  message: string;
  filename: string;
}

// Props interfaces for components
export interface ChatInterfaceProps {
  chatbotId: string;
  onMessageSelect?: (message: Message) => void;
}

export interface GraphViewProps {
  graphData: GraphData;
  onNodeClick?: (node: GraphNode) => void;
  onLinkClick?: (link: GraphLink) => void;
  width?: number;
  height?: number;
}

export interface ChatbotListProps {
  onChatbotSelect?: (chatbot: Chatbot) => void;
  onChatbotDelete?: (chatbotId: string) => void;
}

export interface KnowledgeSourceUploadProps {
  chatbotId: string;
  onUploadComplete?: (filename: string) => void;
  onUploadError?: (error: string) => void;
}

// Environment and configuration types
export interface AppConfig {
  apiBaseUrl: string;
  maxFileSize: number;
  allowedFileTypes: string[];
  wsEndpoint?: string;
}

// Utility types
export type Optional<T, K extends keyof T> = Omit<T, K> & Partial<Pick<T, K>>;
export type RequiredFields<T, K extends keyof T> = T & Required<Pick<T, K>>;