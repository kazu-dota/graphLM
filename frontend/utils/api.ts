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

export const chatWithBot = async (chatbotId: string, query: string) => {
  try {
    const response = await api.post('/api/chat', { chatbot_id: chatbotId, query });
    return response.data;
  } catch (error) {
    console.error('Error chatting with bot:', error);
    throw error;
  }
};
