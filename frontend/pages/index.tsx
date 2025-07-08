import React, { useState, useEffect } from 'react';
import ChatInterface from '../components/ChatInterface';
import GraphPanel from '../components/GraphPanel'; // Import the new GraphPanel
import dynamic from 'next/dynamic';
import { getGraphData } from '../utils/api';
import { ChatbotStatus } from '../components/ChatbotList'; // Import ChatbotStatus
import { Box, AppBar, Toolbar, Typography, Container, Grid, Paper, Accordion, AccordionSummary, AccordionDetails, Card } from '@mui/material'; // Import Material-UI components
import ExpandMoreIcon from '@mui/icons-material/ExpandMore';
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
  const [hoveredNodeId, setHoveredNodeId] = useState<string | null>(null);

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

  const [selectedMessage, setSelectedMessage] = useState<any | null>(null); // State for the selected message

  // New handler to receive reference graph data from ChatInterface
  const handleReferenceGraphData = (graphData: GraphData | null, message: any) => {
    setReferenceGraphData(graphData);
    setSelectedMessage(message);
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
            <Paper sx={{ p: 2 }}>
              <Typography variant="h6" gutterBottom>
                Management
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
                selectedChatbotId={selectedChatbotId} 
                onChatbotsLoaded={setChatbots}
              />
            </Paper>
          </Grid>
          <Grid item xs={12} md={8} lg={9}>
            {selectedChatbotId ? (
              <Box sx={{ display: 'flex', flexDirection: 'column', height: 'calc(100vh - 128px)', gap: 3 }}>
                <Box sx={{ flex: '1 1 auto', minHeight: '200px', resize: 'vertical', overflow: 'auto', border: '1px solid #ccc', p: 1 }}>
                  <ChatInterface 
                    chatbotId={selectedChatbotId} 
                    chatbotStatus={selectedChatbotStatus} 
                    onReferenceDataChange={handleReferenceGraphData} // Pass the new handler
                    onSourceHover={setHoveredNodeId} // Pass the hover handler
                  />
                </Box>
                <Box sx={{ flex: '1 1 auto', minHeight: '200px', resize: 'vertical', overflow: 'auto', border: '1px solid #ccc', p: 1 }}>
                  <GraphPanel 
                    mainGraphData={mainGraphData} 
                    isMainGraphLoading={isGraphLoading} 
                    referenceGraphData={referenceGraphData} 
                    selectedMessage={selectedMessage}
                    hoveredNodeId={hoveredNodeId} // Pass the hovered node ID
                  />
                </Box>
              </Box>
            ) : (
              <Paper sx={{ textAlign: 'center', p: 10, display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', height: '100%' }}>
                <Typography variant="h4" gutterBottom>
                  Welcome to GraphLM
                </Typography>
                <Typography variant="subtitle1">
                  Select a chatbot from the list on the left to start a conversation.
                </Typography>
                <Typography variant="body1" sx={{ mt: 2 }}>
                  Or, create a new chatbot to begin building your own knowledge graph.
                </Typography>
              </Paper>
            )}
          </Grid>
        </Grid>
      </Container>
    </Box>
  );
};

export default Home;
