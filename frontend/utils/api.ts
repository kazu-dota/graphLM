import axios from 'axios';

const API_BASE_URL = process.env.NEXT_PUBLIC_BACKEND_URL || 'http://localhost:8000';

const api = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

export const createChatbot = async (name: string, description?: string) => {
  const formData = new FormData();
  formData.append('name', name);
  if (description) {
    formData.append('description', description);
  }
  try {
    const response = await api.post('/api/chatbots', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    });
    return response.data;
  } catch (error) {
    console.error('Error creating chatbot:', error);
    throw error;
  }
};

export const uploadKnowledgeSource = async (chatbotId: string, file: File) => {
  const formData = new FormData();
  formData.append('file', file);
  try {
    const response = await api.post(`/api/chatbots/${chatbotId}/upload`, formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    });
    return response.data;
  } catch (error) {
    console.error('Error uploading knowledge source:', error);
    throw error;
  }
};

export const fetchChatbots = async () => {
  try {
    const response = await api.get('/api/chatbots');
    return response.data;
  } catch (error) {
    console.error('Error fetching chatbots:', error);
    throw error;
  }
};

export const getIndexingProgress = async (chatbotId: string) => {
  try {
    const response = await api.get(`/api/chatbots/${chatbotId}/indexing_progress`);
    return response.data;
  } catch (error) {
    console.error(`Error fetching indexing progress for chatbot ${chatbotId}:`, error);
    throw error;
  }
};

export const chatWithBot = async (chatbotId: string, query: string, onStreamEvent: (event: any) => void) => {
  try {
    const response = await fetch(`${API_BASE_URL}/api/chat`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Accept': 'text/event-stream',
      },
      body: JSON.stringify({ chatbot_id: chatbotId, query }),
    });

    if (!response.ok) {
      const errorData = await response.json();
      throw new Error(errorData.detail || 'Failed to fetch stream');
    }

    if (!response.body) {
      throw new Error('Response body is null');
    }

    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';

    while (true) {
      const { done, value } = await reader.read();
      if (done) {
        break;
      }
      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split('\n\n');
      buffer = lines.pop() || ''; // Keep the last, possibly incomplete line

      for (const line of lines) {
        if (line.startsWith('data: ')) {
          try {
            const jsonStr = line.substring(6);
            const eventData = JSON.parse(jsonStr);
            onStreamEvent(eventData);
          } catch (e) {
            console.error('Failed to parse stream event:', line, e);
          }
        }
      }
    }

  } catch (error) {
    console.error('Error in chatWithBot:', error);
    onStreamEvent({ event: 'done', data: { error: error instanceof Error ? error.message : String(error) } });
    throw error;
  }
};

export const getGraphData = async (chatbotId: string) => {
  try {
    const response = await api.get(`/api/chatbots/${chatbotId}/graph`);
    return response.data;
  } catch (error) {
    console.error(`Error fetching graph data for chatbot ${chatbotId}:`, error);
    throw error;
  }
};

export const updateChatbot = async (chatbotId: string, name: string, description?: string) => {
  try {
    const response = await api.put(`/api/chatbots/${chatbotId}`, { name, description });
    return response.data;
  } catch (error) {
    console.error(`Error updating chatbot ${chatbotId}:`, error);
    throw error;
  }
};

export const deleteChatbot = async (chatbotId: string) => {
  try {
    await api.delete(`/api/chatbots/${chatbotId}`);
    return true;
  } catch (error) {
    console.error(`Error deleting chatbot ${chatbotId}:`, error);
    throw error;
  }
};
