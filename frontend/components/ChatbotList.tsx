import React, { useEffect, useState, useRef } from 'react';
import { Box, Typography, Card, CardContent, List, ListItem, ListItemText, CircularProgress, Button, Snackbar, Alert, ListItemButton, Chip } from '@mui/material';
import { fetchChatbots, uploadKnowledgeSource, getChatbotStatus } from '../utils/api';

// Define ChatbotStatus enum to match the backend
export enum ChatbotStatus {
  INDEXING = "INDEXING",
  READY = "READY",
  FAILED = "FAILED",
}

interface Chatbot {
  id: string;
  name: string;
  description?: string;
  status: ChatbotStatus;
}

interface ChatbotListProps {
  refresh: boolean;
  onChatbotSelect: (chatbotId: string) => void;
  selectedChatbotId: string | null;
}

const ChatbotList: React.FC<ChatbotListProps> = ({ refresh, onChatbotSelect, selectedChatbotId }) => {
  const [chatbots, setChatbots] = useState<Chatbot[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [notification, setNotification] = useState<{ open: boolean; message: string; severity: 'success' | 'error' }>({ open: false, message: '', severity: 'success' });

  const fileInputRefs = useRef<{[key: string]: HTMLInputElement | null}>({});

  // Function to fetch all chatbots
  const loadChatbots = async () => {
    try {
      setLoading(true);
      const data = await fetchChatbots();
      setChatbots(data);
    } catch (err) {
      setError('Failed to fetch chatbots. Is the backend server running?');
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  // Initial load and refresh
  useEffect(() => {
    loadChatbots();
  }, [refresh]);

  // Polling logic for indexing chatbots
  useEffect(() => {
    const interval = setInterval(() => {
      chatbots.forEach(bot => {
        if (bot.status === ChatbotStatus.INDEXING) {
          getChatbotStatus(bot.id).then(data => {
            if (data.status !== ChatbotStatus.INDEXING) {
              // Refresh the entire list once a bot is ready
              loadChatbots();
            }
          });
        }
      });
    }, 5000); // Poll every 5 seconds

    return () => clearInterval(interval);
  }, [chatbots]);

  const handleFileChange = async (event: React.ChangeEvent<HTMLInputElement>, chatbotId: string) => {
    const file = event.target.files?.[0];
    if (!file) return;

    try {
      await uploadKnowledgeSource(chatbotId, file);
      setNotification({ open: true, message: `File upload started. Indexing will begin shortly.`, severity: 'success' });
      // Immediately update status to INDEXING visually and then reload
      setChatbots(prev => prev.map(b => b.id === chatbotId ? { ...b, status: ChatbotStatus.INDEXING } : b));
      setTimeout(loadChatbots, 1000); // Refresh list after a short delay
    } catch (err) {
      setNotification({ open: true, message: 'File upload failed.', severity: 'error' });
      console.error(err);
    } finally {
      // Reset file input
      const fileInput = fileInputRefs.current[chatbotId];
      if(fileInput) {
        fileInput.value = '';
      }
    }
  };

  const handleCloseNotification = () => {
    setNotification({ ...notification, open: false });
  };

  const getStatusChip = (status: ChatbotStatus) => {
    switch (status) {
      case ChatbotStatus.INDEXING:
        return <Chip label="Indexing..." color="warning" size="small" />;
      case ChatbotStatus.READY:
        return <Chip label="Ready" color="success" size="small" />;
      case ChatbotStatus.FAILED:
        return <Chip label="Failed" color="error" size="small" />;
      default:
        return null;
    }
  };

  return (
    <Card>
      <CardContent>
        <Typography variant="h6" gutterBottom>Existing Chatbots</Typography>
        {loading ? (
          <Box sx={{ display: 'flex', justifyContent: 'center', my: 2 }}>
            <CircularProgress />
          </Box>
        ) : error ? (
          <Typography color="error">{error}</Typography>
        ) : chatbots.length === 0 ? (
          <Typography>No chatbots created yet.</Typography>
        ) : (
          <List>
            {chatbots.map((chatbot) => {
              const isIndexing = chatbot.status === ChatbotStatus.INDEXING;
              return (
                <ListItemButton 
                  key={chatbot.id} 
                  divider 
                  selected={selectedChatbotId === chatbot.id}
                  onClick={() => !isIndexing && onChatbotSelect(chatbot.id)}
                  disabled={isIndexing}
                >
                  <ListItemText
                    primary={chatbot.name}
                    secondary={chatbot.description || 'No description provided.'}
                  />
                  <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                    {getStatusChip(chatbot.status)}
                    <Button
                      variant="contained"
                      component="label"
                      disabled={isIndexing}
                      onClick={(e) => e.stopPropagation()} // Prevent ListItem click event
                    >
                      {isIndexing ? <CircularProgress size={24} /> : 'Upload File'}
                      <input 
                        type="file" 
                        hidden 
                        onChange={(e) => handleFileChange(e, chatbot.id)}
                        ref={el => fileInputRefs.current[chatbot.id] = el}
                      />
                    </Button>
                  </Box>
                </ListItemButton>
              )
            })}
          </List>
        )}
      </CardContent>
      <Snackbar open={notification.open} autoHideDuration={6000} onClose={handleCloseNotification}>
        <Alert onClose={handleCloseNotification} severity={notification.severity} sx={{ width: '100%' }}>
          {notification.message}
        </Alert>
      </Snackbar>
    </Card>
  );
};

export default ChatbotList;