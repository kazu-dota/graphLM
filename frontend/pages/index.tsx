import React, { useState, useEffect } from 'react';
import { AppBar, Toolbar, Typography, Container, Box, Button, Grid } from '@mui/material';
import ChatbotList, { ChatbotStatus } from '../components/ChatbotList';
import ChatbotCreationForm from '../components/ChatbotCreationForm';
import ChatInterface from '../components/ChatInterface';

interface Chatbot {
  id: string;
  name: string;
  description?: string;
  status: ChatbotStatus;
  total_nodes?: number;
  processed_nodes?: number;
  current_step?: string; // IndexingStep is string enum
}

const Home: React.FC = () => {
  const [refreshChatbotList, setRefreshChatbotList] = useState(false);
  const [selectedChatbotId, setSelectedChatbotId] = useState<string | null>(null);
  const [chatbots, setChatbots] = useState<Chatbot[]>([]); // Add state for chatbots
  const [selectedChatbotStatus, setSelectedChatbotStatus] = useState<ChatbotStatus>(ChatbotStatus.READY); // Add state for selected chatbot status

  const handleChatbotCreated = () => {
    setRefreshChatbotList(prev => !prev);
  };

  const handleChatbotSelect = (chatbotId: string) => {
    setSelectedChatbotId(chatbotId);
  };

  // Update selected chatbot status when selectedChatbotId or chatbots list changes
  useEffect(() => {
    if (selectedChatbotId) {
      const currentChatbot = chatbots.find(bot => bot.id === selectedChatbotId);
      if (currentChatbot) {
        setSelectedChatbotStatus(currentChatbot.status);
      }
    }
  }, [selectedChatbotId, chatbots]);

  return (
    <Box sx={{ flexGrow: 1 }}>
      <AppBar position="static">
        <Toolbar>
          <Typography variant="h6" component="div" sx={{ flexGrow: 1 }}>
            GraphLM
          </Typography>
          {/* <Button color="inherit">Login</Button> */}
        </Toolbar>
      </AppBar>
      <Container maxWidth="xl" sx={{ mt: 4, mb: 4 }}>
        <Grid container spacing={3}>
          <Grid item xs={12} md={4}>
            <Box sx={{ display: 'flex', flexDirection: 'column', gap: 3 }}>
              <ChatbotCreationForm onChatbotCreated={handleChatbotCreated} />
              <ChatbotList 
                refresh={refreshChatbotList} 
                onChatbotSelect={handleChatbotSelect} 
                selectedChatbotId={selectedChatbotId} 
                onChatbotsLoaded={setChatbots} // Pass callback to update chatbots state
              />
            </Box>
          </Grid>
          <Grid item xs={12} md={8}>
            {selectedChatbotId ? (
              <ChatInterface chatbotId={selectedChatbotId} chatbotStatus={selectedChatbotStatus} />
            ) : (
              <Box sx={{ textAlign: 'center', mt: 10 }}>
                <Typography variant="h6">Select a chatbot to start chatting</Typography>
              </Box>
            )}
          </Grid>
        </Grid>
      </Container>
    </Box>
  );
};

export default Home;
