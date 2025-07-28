import React, { useState, useEffect } from 'react';
import { Box, AppBar, Toolbar, Typography, Container, Grid, Paper, Accordion, AccordionSummary, AccordionDetails, Fab } from '@mui/material';
import { Help as HelpIcon } from '@mui/icons-material';
import ExpandMoreIcon from '@mui/icons-material/ExpandMore';
import ChatbotCreationForm from '../../components/ChatbotCreationForm';
import ChatbotList from '../../components/ChatbotList';
import OnboardingWizard from '../../components/OnboardingWizard';
import { Chatbot } from '../../types';
import { useRouter } from 'next/router';

const ChatbotsPage: React.FC = () => {
  const [refreshChatbotList, setRefreshChatbotList] = useState(false);
  const [chatbots, setChatbots] = useState<Chatbot[]>([]);
  const [showOnboarding, setShowOnboarding] = useState(false);
  const router = useRouter();

  const handleChatbotCreated = () => {
    setRefreshChatbotList(prev => !prev);
  };

  const handleChatbotSelect = (chatbotId: string) => {
    router.push(`/chat/${chatbotId}`);
  };

  // Check if user is new (no chatbots) and should see onboarding
  useEffect(() => {
    const hasSeenOnboarding = localStorage.getItem('hasSeenOnboarding');
    if (!hasSeenOnboarding && chatbots.length === 0) {
      setShowOnboarding(true);
    }
  }, [chatbots]);

  const handleOnboardingClose = () => {
    setShowOnboarding(false);
  };

  const handleOnboardingComplete = () => {
    localStorage.setItem('hasSeenOnboarding', 'true');
    setShowOnboarding(false);
    // Auto-expand the creation form
    const createAccordion = document.querySelector('[aria-controls="panel1a-content"]') as HTMLElement;
    if (createAccordion && !createAccordion.getAttribute('aria-expanded')) {
      createAccordion.click();
    }
  };

  const handleShowOnboarding = () => {
    setShowOnboarding(true);
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

      {/* Floating Action Button for Help */}
      <Fab
        color="primary"
        aria-label="Show onboarding"
        onClick={handleShowOnboarding}
        sx={{
          position: 'fixed',
          bottom: 16,
          right: 16,
        }}
      >
        <HelpIcon />
      </Fab>

      {/* Onboarding Wizard */}
      <OnboardingWizard
        open={showOnboarding}
        onClose={handleOnboardingClose}
        onComplete={handleOnboardingComplete}
      />
    </Box>
  );
};

export default ChatbotsPage;
