import React, { useState, useEffect } from 'react';
import ChatInterface from '../components/ChatInterface';
import GraphPanel from '../components/GraphPanel'; // Import the new GraphPanel
import dynamic from 'next/dynamic';
import { getGraphData } from '../utils/api';
import { ChatbotStatus } from '../components/ChatbotList'; // Import ChatbotStatus
import { Box, AppBar, Toolbar, Typography, Container, Grid, Paper } from '@mui/material'; // Import Material-UI components
import ChatbotCreationForm from '../components/ChatbotCreationForm'; // Import ChatbotCreationForm
import ChatbotList from '../components/ChatbotList'; // Import ChatbotList

const GraphView = dynamic(() => import('../components/GraphView'), {
  ssr: false,
  loading: () => <p>Loading graph...</p>
});

interface Chatbot {
  id: string;
  name: string;
  description?: string;
  status: ChatbotStatus;
  total_nodes?: number;
  processed_nodes?: number;
  current_step?: string; // IndexingStep is string enum
}

// Define a simple type for graph data
interface GraphData {
  nodes: any[];
  links: any[];
}

const Home: React.FC = () => {
  const [refreshChatbotList, setRefreshChatbotList] = useState(false);
  const [selectedChatbotId, setSelectedChatbotId] = useState<string | null>(null);
  const [chatbots, setChatbots] = useState<Chatbot[]>([]);
  const [selectedChatbotStatus, setSelectedChatbotStatus] = useState<ChatbotStatus>(ChatbotStatus.READY);
  const [mainGraphData, setMainGraphData] = useState<GraphData | null>(null);
  const [isGraphLoading, setIsGraphLoading] = useState(false);
  const [referenceGraphData, setReferenceGraphData] = useState<GraphData | null>(null); // New state for reference graph

  const handleChatbotCreated = () => {
    setRefreshChatbotList(prev => !prev);
  };

  const handleChatbotSelect = async (chatbotId: string) => {
    setSelectedChatbotId(chatbotId);
    // Reset previous graph data and reference graph data
    setMainGraphData(null);
    setReferenceGraphData(null);

    // Fetch new graph data
    if (chatbotId) {
      setIsGraphLoading(true);
      try {
        const data = await getGraphData(chatbotId);
        setMainGraphData(data);
      } catch (error) {
        console.error("Failed to fetch graph data", error);
        setMainGraphData(null); // Set to null on error
      } finally {
        setIsGraphLoading(false);
      }
    }
  };

  // New handler to receive reference graph data from ChatInterface
  const handleReferenceGraphData = (graphData: GraphData | null) => {
    setReferenceGraphData(graphData);
  };

  useEffect(() => {
    if (selectedChatbotId) {
      const currentChatbot = chatbots.find(bot => bot.id === selectedChatbotId);
      if (currentChatbot) {
        setSelectedChatbotStatus(currentChatbot.status);
        // Fetch graph data if the bot is ready
        if (currentChatbot.status === ChatbotStatus.READY) {
          handleChatbotSelect(selectedChatbotId);
        }
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
        </Toolbar>
      </AppBar>
      <Container maxWidth="xl" sx={{ mt: 4, mb: 4 }}>
        <Grid container spacing={3}>
          <Grid item xs={12} md={4} lg={3}>
            <Box sx={{ display: 'flex', flexDirection: 'column', gap: 3 }}>
              <ChatbotCreationForm onChatbotCreated={handleChatbotCreated} />
              <ChatbotList 
                refresh={refreshChatbotList} 
                onChatbotSelect={handleChatbotSelect} 
                selectedChatbotId={selectedChatbotId} 
                onChatbotsLoaded={setChatbots}
              />
            </Box>
          </Grid>
          <Grid item xs={12} md={8} lg={9}>
            {selectedChatbotId ? (
              <Grid container spacing={2}>
                <Grid item xs={12} lg={6}>
                  <ChatInterface 
                    chatbotId={selectedChatbotId} 
                    chatbotStatus={selectedChatbotStatus} 
                    onReferenceGraphData={handleReferenceGraphData} // Pass the new handler
                  />
                </Grid>
                <Grid item xs={12} lg={6}>
                  <GraphPanel 
                    mainGraphData={mainGraphData} 
                    isMainGraphLoading={isGraphLoading} 
                    referenceGraphData={referenceGraphData} 
                  />
                </Grid>
              </Grid>
            ) : (
              <Paper sx={{ textAlign: 'center', p: 10 }}>
                <Typography variant="h6">Select a chatbot to get started</Typography>
                <Typography variant="body1">Once you select a chatbot, you can start a conversation and view its knowledge graph here.</Typography>
              </Paper>
            )}
          </Grid>
        </Grid>
      </Container>
    </Box>
  );
};

export default Home;
