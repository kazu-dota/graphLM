import React, { useState, useEffect, useRef } from 'react';
import { Box, Typography, TextField, Button, Paper, CircularProgress, List, ListItem, ListItemText, Divider, Grid } from '@mui/material';
import { chatWithBot } from '../utils/api';
import dynamic from 'next/dynamic';

const GraphView = dynamic(() => import('./GraphView'), { 
  ssr: false, 
  loading: () => <p>Loading graph...</p> 
});

interface Message {
  sender: 'user' | 'bot';
  text: string;
  sources?: any[];
  graphData?: any; // NEW: Add graph data to message
}

import { ChatbotStatus } from './ChatbotList'; // Import the enum

interface ChatInterfaceProps {
  chatbotId: string;
  chatbotStatus: ChatbotStatus;
  onReferenceGraphData: (graphData: any) => void; // New prop to pass graph data to parent
}

const ChatInterface: React.FC<ChatInterfaceProps> = ({ chatbotId, chatbotStatus, onReferenceGraphData }) => {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  // Clear messages when chatbot changes
  useEffect(() => {
    setMessages([]);
    onReferenceGraphData(null);
  }, [chatbotId]);

  const handleSend = async () => {
    if (!input.trim() || chatbotStatus !== ChatbotStatus.READY) return;

    setLoading(true);
    setInput('');
    try {
      const response = await chatWithBot(chatbotId, input);
      setMessages(prev => [
        ...prev,
        { sender: 'user', text: input },
        { sender: 'bot', text: response.response, sources: response.sources }
      ]);
      onReferenceGraphData(response.graph_data);
    } catch (error: any) {
      const detail = error.response?.data?.detail || 'Sorry, something went wrong.';
      setMessages(prev => [
        ...prev,
        { sender: 'user', text: input },
        { sender: 'bot', text: detail }
      ]);
      onReferenceGraphData(null);
    } finally {
      setLoading(false);
    }
  };

  const isChatDisabled = chatbotStatus !== ChatbotStatus.READY;

  return (
    <Paper elevation={3} sx={{ height: '75vh', display: 'flex', flexDirection: 'column' }}>
      <Typography variant="h6" sx={{ p: 2, borderBottom: '1px solid #ddd' }}>
        Chat with Bot
      </Typography>
      <Box sx={{ flexGrow: 1, overflowY: 'auto', p: 2 }}>
        {messages.map((msg, index) => {
          console.log(`Rendering message ${index}:`, msg);
          return (
            <Box key={index} sx={{ mb: 2, textAlign: msg.sender === 'user' ? 'right' : 'left' }}>
              <Typography variant="body1" sx={{ p: 1.5, display: 'inline-block', maxWidth: '70%', bgcolor: msg.sender === 'user' ? 'primary.main' : 'grey.300', color: msg.sender === 'user' ? 'white' : 'black' }}>
                {msg.text}
              </Typography>
              {msg.sender === 'bot' && msg.sources && msg.sources.length > 0 && (
                <Box sx={{ mt: 1 }}>
                  <Typography variant="subtitle2">Sources:</Typography>
                  <List dense>
                    {msg.sources.map((source, i) => (
                      <ListItem key={i}>
                        <ListItemText primary={source.document_name} secondary={`Page: ${source.page_number}`} />
                      </ListItem>
                    ))}
                  </List>
                </Box>
              )}
            </Box>
          );
        })}
        {loading && <CircularProgress sx={{ display: 'block', mx: 'auto' }} />}
        <div ref={messagesEndRef} />
      </Box>
      <Divider />
      <Box sx={{ p: 2, display: 'flex', gap: 1 }}>
        <TextField 
          fullWidth 
          variant="outlined" 
          placeholder={isChatDisabled ? "Please wait for indexing to complete..." : "Type your message..."} 
          value={input} 
          onChange={(e) => setInput(e.target.value)}
          onKeyPress={(e) => e.key === 'Enter' && !isChatDisabled && handleSend()}
          disabled={isChatDisabled || loading}
        />
        <Button variant="contained" onClick={handleSend} disabled={isChatDisabled || loading}>
          Send
        </Button>
      </Box>
    </Paper>
  );
};

export default ChatInterface;
