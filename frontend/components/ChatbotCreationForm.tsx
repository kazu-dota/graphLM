import React, { useState } from 'react';
import { Box, Typography, TextField, Button, Card, CardContent } from '@mui/material';
import { createChatbot } from '../utils/api';

interface ChatbotCreationFormProps {
  onChatbotCreated: () => void;
}

const ChatbotCreationForm: React.FC<ChatbotCreationFormProps> = ({ onChatbotCreated }) => {
  const [name, setName] = useState('');
  const [description, setDescription] = useState('');

  const handleSubmit = async (event: React.FormEvent) => {
    event.preventDefault();
    try {
      await createChatbot(name, description);
      alert('Chatbot created successfully!');
      setName('');
      setDescription('');
      onChatbotCreated(); // Notify parent component to refresh chatbot list
    } catch (error) {
      alert('Failed to create chatbot.');
    }
  };

  return (
    <Card>
      <CardContent>
        <Typography variant="h6" gutterBottom>Create New Chatbot</Typography>
        <Box component="form" onSubmit={handleSubmit} sx={{ mt: 2 }}>
          <TextField
            label="Chatbot Name"
            variant="outlined"
            fullWidth
            value={name}
            onChange={(e) => setName(e.target.value)}
            required
            sx={{ mb: 2 }}
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
          />
          <Button type="submit" variant="contained" color="primary">
            Create Chatbot
          </Button>
        </Box>
      </CardContent>
    </Card>
  );
};

export default ChatbotCreationForm;