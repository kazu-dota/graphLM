import React, { useEffect, useState, useRef } from 'react';
import { Box, Typography, Card, CardContent, List, ListItem, ListItemText, CircularProgress, Button, Snackbar, Alert, ListItemButton, Chip } from '@mui/material';
import { fetchChatbots, uploadKnowledgeSource, getIndexingProgress } from '../utils/api';

// Define ChatbotStatus enum to match the backend
export enum ChatbotStatus {
  INDEXING = "INDEXING",
  READY = "READY",
  FAILED = "FAILED",
}

export enum IndexingStep {
  LOADING_DOCUMENTS = "Loading Documents",
  PARSING_NODES = "Parsing Nodes",
  GENERATING_EMBEDDINGS = "Generating Embeddings",
  BUILDING_GRAPH = "Building Graph",
}

interface Chatbot {
  id: string;
  name: string;
  description?: string;
  status: ChatbotStatus;
  total_nodes?: number;
  processed_nodes?: number;
  current_step?: IndexingStep;
}

interface ChatbotListProps {
  refresh: boolean;
  onChatbotSelect: (chatbotId: string) => void;
  selectedChatbotId: string | null;
  onChatbotsLoaded: (chatbots: Chatbot[]) => void; // New prop
}

const ChatbotList: React.FC<ChatbotListProps> = ({ refresh, onChatbotSelect, selectedChatbotId, onChatbotsLoaded }) => {
  const [chatbots, setChatbots] = useState<Chatbot[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [notification, setNotification] = useState<{ open: boolean; message: string; severity: 'success' | 'error' }>({ open: false, message: '', severity: 'success' });

  // Function to fetch all chatbots
  const loadChatbots = async () => {
    try {
      setLoading(true);
      const data = await fetchChatbots();
      setChatbots(data);
      onChatbotsLoaded(data); // Call the new prop
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
      chatbots.forEach(async bot => {
        if (bot.status === ChatbotStatus.INDEXING) {
          try {
            const data = await getIndexingProgress(bot.id);
            setChatbots(prev => prev.map(b => 
              b.id === bot.id ? { ...b, status: data.status, total_nodes: data.total_nodes, processed_nodes: data.processed_nodes, current_step: data.current_step } : b
            ));
            if (data.status !== ChatbotStatus.INDEXING) {
              // Refresh the entire list once a bot is ready or failed
              loadChatbots();
            }
          } catch (err) {
            console.error(`Failed to fetch indexing progress for ${bot.id}:`, err);
            // Optionally, set bot status to FAILED if polling consistently fails
            setChatbots(prev => prev.map(b => b.id === bot.id ? { ...b, status: ChatbotStatus.FAILED } : b));
          }
        }
      });
    }, 2000); // Poll every 2 seconds for more granular updates

    return () => clearInterval(interval);
  }, [chatbots]);

  const handleCloseNotification = () => {
    setNotification({ ...notification, open: false });
  };

  const getStatusChip = (chatbot: Chatbot) => {
    switch (chatbot.status) {
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
              const progress = chatbot.total_nodes && chatbot.total_nodes > 0 ? Math.round((chatbot.processed_nodes / chatbot.total_nodes) * 100) : 0;

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
                    secondary={
                      <React.Fragment>
                        <Typography
                          sx={{ display: 'inline' }}
                          component="span"
                          variant="body2"
                          color="text.primary"
                        >
                          {chatbot.description || 'No description provided.'}
                        </Typography>
                        {isIndexing && (
                          <Box sx={{ width: '100%', mt: 1 }}>
                            <LinearProgress variant="determinate" value={progress} />
                            <Typography variant="caption" color="text.secondary">
                              {chatbot.current_step || 'Starting...'}: {chatbot.processed_nodes || 0}/{chatbot.total_nodes || 0} nodes ({progress}%)
                            </Typography>
                          </Box>
                        )}
                      </React.Fragment>
                    }
                  />
                  <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                    {!isIndexing && getStatusChip(chatbot)}
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