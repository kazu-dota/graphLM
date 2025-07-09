import React, { useState } from 'react';
import { Box, AppBar, Toolbar, Typography, Container, Grid, Paper, Accordion, AccordionSummary, AccordionDetails } from '@mui/material';
import ExpandMoreIcon from '@mui/icons-material/ExpandMore';
import ChatbotCreationForm from '../../components/ChatbotCreationForm';
import ChatbotList from '../../components/ChatbotList';
import { useRouter } from 'next/router';

const ChatbotsPage: React.FC = () => {
  const [refreshChatbotList, setRefreshChatbotList] = useState(false);
  const [chatbots, setChatbots] = useState<any[]>([]); // Use any[] for now, refine later if needed
  const router = useRouter();

  const handleChatbotCreated = () => {
    setRefreshChatbotList(prev => !prev);
  };

  const handleChatbotSelect = (chatbotId: string) => {
    router.push(`/chat/${chatbotId}`);
  };

  return (
    <Box sx={{ flexGrow: 1 }}>
      <AppBar position="static">
        <Toolbar>
          <Typography variant="h6" component="div" sx={{ flexGrow: 1 }}>
            GraphLM - Chatbot Management
          </Typography>
        </Toolbar>
      </AppBar>
      <Container maxWidth="xl" sx={{ mt: 4, mb: 4 }}>
        <Grid container spacing={3}>
          <Grid item xs={12}>
            <Paper sx={{ p: 2 }}>
              <Typography variant="h6" gutterBottom>
                Chatbot Management
              </Typography>
              <Accordion defaultExpanded>
                <AccordionSummary
                  expandIcon={<ExpandMoreIcon />}
                  aria-controls="panel1a-content"
                  id="panel1a-header"
                >
                  <Typography>Create New Chatbot</Typography>
                </AccordionSummary>
                <AccordionDetails>
                  <ChatbotCreationForm onChatbotCreated={handleChatbotCreated} />
                </AccordionDetails>
              </Accordion>
              <ChatbotList 
                refresh={refreshChatbotList} 
                onChatbotSelect={handleChatbotSelect} 
                selectedChatbotId={null} // No selected chatbot on this page
                onChatbotsLoaded={setChatbots}
              />
            </Paper>
          </Grid>
        </Grid>
      </Container>
    </Box>
  );
};

export default ChatbotsPage;
