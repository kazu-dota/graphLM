import React, { useEffect, useState, useRef } from 'react';
import { Box, Typography, Card, CardContent, List, ListItem, ListItemText, CircularProgress, Button, Snackbar, Alert, ListItemButton, Chip, IconButton, LinearProgress } from '@mui/material';
import { fetchChatbots, updateChatbot, deleteChatbot, getIndexingProgress } from '../utils/api';
import EditIcon from '@mui/icons-material/Edit';
import DeleteIcon from '@mui/icons-material/Delete';
import Dialog from '@mui/material/Dialog';
import DialogActions from '@mui/material/DialogActions';
import DialogContent from '@mui/material/DialogContent';
import DialogContentText from '@mui/material/DialogContentText';
import DialogTitle from '@mui/material/DialogTitle';
import TextField from '@mui/material/TextField';

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

  const [openEditDialog, setOpenEditDialog] = useState(false);
  const [currentEditChatbot, setCurrentEditChatbot] = useState<Chatbot | null>(null);
  const [editName, setEditName] = useState('');
  const [editDescription, setEditDescription] = useState('');

  const [openDeleteConfirm, setOpenDeleteConfirm] = useState(false);
  const [chatbotToDelete, setChatbotToDelete] = useState<Chatbot | null>(null);

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

  const handleEditClick = (chatbot: Chatbot) => {
    setCurrentEditChatbot(chatbot);
    setEditName(chatbot.name);
    setEditDescription(chatbot.description || '');
    setOpenEditDialog(true);
  };

  const handleEditDialogClose = () => {
    setOpenEditDialog(false);
    setCurrentEditChatbot(null);
  };

  const handleUpdateChatbot = async () => {
    if (!currentEditChatbot) return;
    try {
      await updateChatbot(currentEditChatbot.id, editName, editDescription);
      setNotification({ open: true, message: 'Chatbot updated successfully!', severity: 'success' });
      loadChatbots();
      handleEditDialogClose();
    } catch (err) {
      setNotification({ open: true, message: 'Failed to update chatbot.', severity: 'error' });
      console.error(err);
    }
  };

  const handleDeleteClick = (chatbot: Chatbot) => {
    setChatbotToDelete(chatbot);
    setOpenDeleteConfirm(true);
  };

  const handleDeleteConfirmClose = () => {
    setOpenDeleteConfirm(false);
    setChatbotToDelete(null);
  };

  const handleDeleteChatbot = async () => {
    if (!chatbotToDelete) return;
    try {
      await deleteChatbot(chatbotToDelete.id);
      setNotification({ open: true, message: 'Chatbot deleted successfully!', severity: 'success' });
      loadChatbots();
      handleDeleteConfirmClose();
      if (selectedChatbotId === chatbotToDelete.id) {
        onChatbotSelect(''); // Deselect if the deleted chatbot was selected
      }
    } catch (err) {
      setNotification({ open: true, message: 'Failed to delete chatbot.', severity: 'error' });
      console.error(err);
    }
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
                        {chatbot.status === ChatbotStatus.FAILED && chatbot.description && (
                          <Typography variant="caption" color="error" sx={{ display: 'block', mt: 0.5 }}>
                            Error: {chatbot.description}
                          </Typography>
                        )}
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
                    <IconButton edge="end" aria-label="edit" onClick={(e) => {
                      e.stopPropagation();
                      handleEditClick(chatbot);
                    }} disabled={isIndexing}>
                      <EditIcon fontSize="small" />
                    </IconButton>
                    <IconButton edge="end" aria-label="delete" onClick={(e) => {
                      e.stopPropagation();
                      handleDeleteClick(chatbot);
                    }} disabled={isIndexing}>
                      <DeleteIcon fontSize="small" />
                    </IconButton>
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

      {/* Edit Chatbot Dialog */}
      <Dialog open={openEditDialog} onClose={handleEditDialogClose}>
        <DialogTitle>Edit Chatbot</DialogTitle>
        <DialogContent>
          <TextField
            autoFocus
            margin="dense"
            id="name"
            label="Chatbot Name"
            type="text"
            fullWidth
            variant="standard"
            value={editName}
            onChange={(e) => setEditName(e.target.value)}
            sx={{ mb: 2 }}
          />
          <TextField
            margin="dense"
            id="description"
            label="Description"
            type="text"
            fullWidth
            multiline
            rows={4}
            variant="standard"
            value={editDescription}
            onChange={(e) => setEditDescription(e.target.value)}
          />
        </DialogContent>
        <DialogActions>
          <Button onClick={handleEditDialogClose}>Cancel</Button>
          <Button onClick={handleUpdateChatbot}>Save</Button>
        </DialogActions>
      </Dialog>

      {/* Delete Confirmation Dialog */}
      <Dialog open={openDeleteConfirm} onClose={handleDeleteConfirmClose}>
        <DialogTitle>Confirm Deletion</DialogTitle>
        <DialogContent>
          <DialogContentText>
            Are you sure you want to delete "{chatbotToDelete?.name}"? This action cannot be undone.
          </DialogContentText>
        </DialogContent>
        <DialogActions>
          <Button onClick={handleDeleteConfirmClose}>Cancel</Button>
          <Button onClick={handleDeleteChatbot} color="error">Delete</Button>
        </DialogActions>
      </Dialog>
    </Card>
  );
};

export default ChatbotList;