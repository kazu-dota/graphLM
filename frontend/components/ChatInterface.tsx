import React, { useState } from 'react';
import { Box, Typography, TextField, Button, Paper, CircularProgress, List, ListItem, ListItemText, Divider } from '@mui/material';
import { chatWithBot } from '../utils/api';

interface Message {
  sender: 'user' | 'bot';
  text: string;
  sources?: any[];
}

import { ChatbotStatus } from './ChatbotList'; // Import the enum

interface ChatInterfaceProps {
  chatbotId: string;
  chatbotStatus: ChatbotStatus;
}

const ChatInterface: React.FC<ChatInterfaceProps> = ({ chatbotId, chatbotStatus }) => {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);

  // Clear messages when chatbot changes
  useEffect(() => {
    setMessages([]);
  }, [chatbotId]);

  const handleSend = async () => {
    if (!input.trim() || chatbotStatus !== ChatbotStatus.READY) return;

    const userMessage: Message = { sender: 'user', text: input };
    setMessages(prev => [...prev, userMessage]);
    setInput('');
    setLoading(true);

    try {
      const response = await chatWithBot(chatbotId, input);
      const botMessage: Message = { sender: 'bot', text: response.response, sources: response.sources };
      setMessages(prev => [...prev, botMessage]);
    } catch (error: any) {
      const detail = error.response?.data?.detail || 'Sorry, something went wrong.';
      const errorMessage: Message = { sender: 'bot', text: detail };
      setMessages(prev => [...prev, errorMessage]);
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
        {messages.map((msg, index) => (
          <Box key={index} sx={{ mb: 2, textAlign: msg.sender === 'user' ? 'right' : 'left' }}>
            <Paper elevation={1} sx={{ p: 1.5, display: 'inline-block', maxWidth: '70%', bgcolor: msg.sender === 'user' ? 'primary.main' : 'grey.300', color: msg.sender === 'user' ? 'white' : 'black' }}>
              <Typography variant="body1">{msg.text}</Typography>
            </Paper>
            {msg.sender === 'bot' && msg.sources && (
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
        ))}
        {loading && <CircularProgress sx={{ display: 'block', mx: 'auto' }} />}
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
