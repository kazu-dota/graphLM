import React, { useState, useCallback } from 'react';
import { Box, Typography, TextField, Button, Paper, List, ListItem, ListItemText, IconButton, LinearProgress, ListItemIcon } from '@mui/material';
import { useDropzone } from 'react-dropzone';
import { createChatbot, uploadKnowledgeSource } from '../utils/api';
import CloudUploadIcon from '@mui/icons-material/CloudUpload';
import DeleteIcon from '@mui/icons-material/Delete';
import InsertDriveFileIcon from '@mui/icons-material/InsertDriveFile';

interface ChatbotCreationFormProps {
  onChatbotCreated: () => void;
}

const ChatbotCreationForm: React.FC<ChatbotCreationFormProps> = ({ onChatbotCreated }) => {
  const [name, setName] = useState('');
  const [description, setDescription] = useState('');
  const [files, setFiles] = useState<File[]>([]);
  const [isCreating, setIsCreating] = useState(false);

  const onDrop = useCallback((acceptedFiles: File[]) => {
    setFiles(prevFiles => [...prevFiles, ...acceptedFiles]);
  }, []);

  const removeFile = (fileToRemove: File) => {
    setFiles(prevFiles => prevFiles.filter(file => file !== fileToRemove));
  };

  const { getRootProps, getInputProps, isDragActive } = useDropzone({ onDrop });

  const handleReset = () => {
    setName('');
    setDescription('');
    setFiles([]);
    setIsCreating(false);
  };

  const handleCreateAndBuild = async () => {
    if (!name.trim() || files.length === 0) {
      alert('Please provide a name and at least one file.');
      return;
    }
    setIsCreating(true);
    try {
      // Step 1: Create the chatbot entry
      const chatbot = await createChatbot(name, description);
      
      // Step 2: Upload files for the new chatbot
      // This could be improved with parallel uploads
      for (const file of files) {
        await uploadKnowledgeSource(chatbot.id, file);
      }
      
      alert(`Chatbot "${name}" created and file upload started. Indexing will begin shortly.`);
      onChatbotCreated(); // Refresh the list in the parent component
      handleReset(); // Clear the form for the next creation

    } catch (error) {
      alert('An error occurred during chatbot creation or file upload.');
      console.error(error);
    } finally {
      setIsCreating(false);
    }
  };

  return (
    <Box sx={{ width: '100%' }}>
      <Typography variant="h6" gutterBottom>Create New Chatbot</Typography>
      <TextField
        label="Chatbot Name"
        variant="outlined"
        fullWidth
        value={name}
        onChange={(e) => setName(e.target.value)}
        required
        sx={{ mt: 1, mb: 2 }}
        disabled={isCreating}
      />
      <TextField
        label="Description (Optional)"
        variant="outlined"
        fullWidth
        multiline
        rows={3}
        value={description}
        onChange={(e) => setDescription(e.target.value)}
        sx={{ mb: 2 }}
        disabled={isCreating}
      />

      <Paper
        {...getRootProps()}
        sx={{
          p: 4,
          border: '2px dashed',
          borderColor: isDragActive ? 'primary.main' : 'divider',
          textAlign: 'center',
          cursor: 'pointer',
          mb: 2,
          backgroundColor: isDragActive ? 'action.hover' : 'transparent',
          color: 'text.secondary'
        }}
      >
        <input {...getInputProps()} />
        <CloudUploadIcon sx={{ fontSize: 48, color: 'text.secondary' }} />
        <Typography>Drag & drop knowledge files here, or click to select</Typography>
      </Paper>

      {files.length > 0 && (
        <Box sx={{ mb: 2 }}>
          <Typography variant="subtitle1">Files to upload:</Typography>
          <List dense>
            {files.map((file, index) => (
              <ListItem key={index} secondaryAction={
                <IconButton edge="end" aria-label="delete" onClick={() => removeFile(file)} disabled={isCreating}>
                  <DeleteIcon />
                </IconButton>
              }>
                <ListItemIcon>
                  <InsertDriveFileIcon />
                </ListItemIcon>
                <ListItemText primary={file.name} secondary={`${(file.size / 1024).toFixed(2)} KB`} />
              </ListItem>
            ))}
          </List>
        </Box>
      )}

      <Button 
        variant="contained" 
        onClick={handleCreateAndBuild} 
        disabled={isCreating || !name.trim() || files.length === 0}
        fullWidth
        sx={{ mt: 2, py: 1.5 }}
      >
        {isCreating ? 'Creating...' : 'Create and Build Chatbot'}
      </Button>
      {isCreating && <LinearProgress sx={{ mt: 1 }} />}
    </Box>
  );
};

export default ChatbotCreationForm;