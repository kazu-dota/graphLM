import React, { useState, useEffect, useRef } from 'react';
import { Box, Typography, TextField, Button, Paper, CircularProgress, List, ListItem, ListItemText, Divider, Grid, Card, CardContent } from '@mui/material';
import { chatWithBot } from '../utils/api';
import { ChatbotStatus } from './ChatbotList';

interface Message {
  id: number;
  sender: 'user' | 'bot';
  text: string;
  sources?: any[];
  graphData?: any;
  status?: 'pending' | 'completed' | 'error'; // Add status property
}

interface ChatInterfaceProps {
  chatbotId: string;
  chatbotStatus: ChatbotStatus;
  onReferenceDataChange: (graphData: any, message: Message | null) => void;
}

const ChatInterface: React.FC<ChatInterfaceProps> = ({ chatbotId, chatbotStatus, onReferenceDataChange }) => {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const [selectedMessage, setSelectedMessage] = useState<Message | null>(null);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  useEffect(() => {
    setMessages([]);
    setSelectedMessage(null);
    onReferenceDataChange(null, null);
  }, [chatbotId]);

  const handleSend = async () => {
    if (!input.trim() || chatbotStatus !== ChatbotStatus.READY || loading) return;

    const userMessage: Message = {
      id: Date.now(),
      sender: 'user',
      text: input,
    };

    const pendingBotMessageId = Date.now() + 1;
    const pendingBotMessage: Message = {
      id: pendingBotMessageId,
      sender: 'bot',
      text: '...', // Placeholder text
      status: 'pending',
    };
    
    // Add both user message and pending bot message
    setMessages(prev => [...prev, userMessage, pendingBotMessage]);
    
    const currentInput = input;
    setInput('');
    setLoading(true);

    try {
      const response = await chatWithBot(chatbotId, currentInput);
      const completedBotMessage: Message = {
        id: pendingBotMessageId, // Use the same ID to update the pending message
        sender: 'bot',
        text: response.response,
        sources: response.sources,
        graphData: response.graph_data,
        status: 'completed',
      };
      
      // Update the pending bot message with the completed one
      setMessages(prev => prev.map(msg => 
        msg.id === pendingBotMessageId ? completedBotMessage : msg
      ));
      setSelectedMessage(completedBotMessage);
      onReferenceDataChange(completedBotMessage.graphData, completedBotMessage);

    } catch (error: any) {
      const detail = error.response?.data?.detail || 'Sorry, something went wrong.';
      const errorBotMessage: Message = {
        id: pendingBotMessageId,
        sender: 'bot',
        text: `Error: ${detail}`,
        status: 'error',
      };
      
      // Update the pending bot message with the error one
      setMessages(prev => prev.map(msg => 
        msg.id === pendingBotMessageId ? errorBotMessage : msg
      ));
      setSelectedMessage(errorBotMessage);
      onReferenceDataChange(null, errorBotMessage);
    } finally {
      setLoading(false);
    }
  };

  const handleSelectMessage = (message: Message) => {
    if (message.sender === 'bot' && message.status === 'completed') { // Only select completed bot messages
      setSelectedMessage(message);
      onReferenceDataChange(message.graphData, message);
    }
  };

  const isChatDisabled = chatbotStatus !== ChatbotStatus.READY;

  return (
    <Box sx={{ width: '100%', height: '100%' }}>
      <Paper elevation={3} sx={{ height: '100%', display: 'flex', flexDirection: 'column' }}>
        <Typography variant="h6" sx={{ p: 2, borderBottom: '1px solid #ddd' }}>
          Conversation
        </Typography>
        <Box sx={{ flexGrow: 1, overflowY: 'auto', p: 2 }}>
          {messages.map((msg) => (
            <Box key={msg.id} sx={{ mb: 2, cursor: msg.sender === 'bot' && msg.status === 'completed' ? 'pointer' : 'default' }} onClick={() => handleSelectMessage(msg)}>
              <Paper
                elevation={selectedMessage?.id === msg.id ? 4 : 1}
                sx={{
                  p: 1.5,
                  display: 'inline-block',
                  minWidth: '10%',
                  maxWidth: '90%',
                  bgcolor: msg.sender === 'user' ? 'primary.light' : 'background.paper',
                  border: '1px solid',
                  borderColor: selectedMessage?.id === msg.id ? 'primary.dark' : '#ddd',
                  boxShadow: selectedMessage?.id === msg.id ? '0px 0px 8px rgba(0, 0, 0, 0.2)' : 'none',
                  float: msg.sender === 'user' ? 'right' : 'left',
                  clear: 'both',
                }}
              >
                <Typography variant="body1" sx={{ wordWrap: 'break-word' }}>
                  {msg.text}
                  {msg.sender === 'bot' && msg.status === 'pending' && (
                    <CircularProgress size={16} sx={{ ml: 1 }} />
                  )}
                </Typography>
                {msg.sender === 'bot' && msg.sources && msg.sources.length > 0 && (
                  <Box sx={{ mt: 2 }}>
                    <Divider />
                    <Typography variant="subtitle2" sx={{ mt: 1, mb: 1 }}>
                      References:
                    </Typography>
                    <List dense>
                      {msg.sources.filter(source => source.document_name).map((source, i) => (
                        <ListItem key={i} sx={{ pl: 0 }}>
                          <ListItemText
                            primary={source.document_name}
                            secondary={`Page: ${source.page_number || 'N/A'}`}
                          />
                        </ListItem>
                      ))}
                    </List>
                  </Box>
                )}
              </Paper>
            </Box>
          ))}
          <div ref={messagesEndRef} />
        </Box>
        <Divider />
        <Box sx={{ p: 2, display: 'flex', alignItems: 'center', gap: 1 }}>
          <TextField
            fullWidth
            variant="outlined"
            placeholder={isChatDisabled ? "Indexing is in progress..." : "Type your message..."}
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyPress={(e) => {
              if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                handleSend();
              }
            }}
            disabled={isChatDisabled || loading}
            multiline
            rows={3}
          />
          <Box sx={{ position: 'relative' }}>
            <Button variant="contained" onClick={handleSend} disabled={isChatDisabled || loading}>
              Send
            </Button>
            {loading && (
              <CircularProgress
                size={24}
                sx={{
                  position: 'absolute',
                  top: '50%',
                  left: '50%',
                  marginTop: '-12px',
                  marginLeft: '-12px',
                }}
              />
            )}
          </Box>
        </Box>
      </Paper>
    </Box>
  );
};

export default ChatInterface;