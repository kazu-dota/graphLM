import React, { useState, useEffect } from 'react';
import ChatInterface from '../../components/ChatInterface';
import GraphPanel from '../../components/GraphPanel';
import dynamic from 'next/dynamic';
import { getGraphData } from '../../utils/api';
import { ChatbotStatus } from '../../components/ChatbotList';
import { Box, AppBar, Toolbar, Typography, Container, Paper } from '@mui/material';
import { useRouter } from 'next/router';

const GraphView = dynamic(() => import('../../components/GraphView'), {
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
  current_step?: string;
}

interface GraphData {
  nodes: any[];
  links: any[];
}

const ChatPage: React.FC = () => {
  const router = useRouter();
  const { id: chatbotId } = router.query; // Get chatbotId from query parameter

  const [selectedChatbotStatus, setSelectedChatbotStatus] = useState<ChatbotStatus>(ChatbotStatus.READY);
  const [mainGraphData, setMainGraphData] = useState<GraphData | null>(null);
  const [isGraphLoading, setIsGraphLoading] = useState(false);
  const [referenceGraphData, setReferenceGraphData] = useState<GraphData | null>(null);
  const [hoveredNodeId, setHoveredNodeId] = useState<string | null>(null);
  const [selectedMessage, setSelectedMessage] = useState<any | null>(null);

  useEffect(() => {
    if (chatbotId && typeof chatbotId === 'string') {
      // In a real app, you'd fetch the chatbot details here to get its status
      // For now, we'll assume it's ready for chat if an ID is provided.
      // You might want to add a fetchChatbotById API call here.
      setSelectedChatbotStatus(ChatbotStatus.READY); 
      
      // Fetch graph data for the selected chatbot
      setIsGraphLoading(true);
      getGraphData(chatbotId as string)
        .then(data => {
          setMainGraphData(data);
        })
        .catch(error => {
          console.error("Failed to fetch graph data", error);
          setMainGraphData(null);
        })
        .finally(() => {
          setIsGraphLoading(false);
        });
    } else {
      // If no chatbotId, redirect to the chatbot selection page
      router.push('/chatbots');
    }
  }, [chatbotId, router]);

  const handleReferenceGraphData = (graphData: GraphData | null, message: any) => {
    setReferenceGraphData(graphData);
    setSelectedMessage(message);
  };

  return (
    <Box sx={{ flexGrow: 1 }}>
      <AppBar position="static">
        <Toolbar>
          <Typography variant="h6" component="div" sx={{ flexGrow: 1 }}>
            GraphLM - Chat with Chatbot
          </Typography>
        </Toolbar>
      </AppBar>
      <Container maxWidth="xl" sx={{ mt: 4, mb: 4 }}>
        {chatbotId ? (
          <Box sx={{ display: 'flex', flexDirection: 'column', height: 'calc(100vh - 128px)', gap: 3 }}>
            <Box sx={{ flex: '2 1 auto', minHeight: '500px', resize: 'vertical', overflow: 'auto', border: 1, borderColor: 'divider', p: 1 }}>
              <ChatInterface 
                chatbotId={chatbotId as string} 
                chatbotStatus={selectedChatbotStatus} 
                onReferenceDataChange={handleReferenceGraphData}
                onSourceHover={setHoveredNodeId}
              />
            </Box>
            <Box sx={{ flex: '1 1 auto', minHeight: '200px', resize: 'vertical', overflow: 'auto', border: 1, borderColor: 'divider', p: 1 }}>
              <GraphPanel 
                mainGraphData={mainGraphData} 
                isMainGraphLoading={isGraphLoading} 
                referenceGraphData={referenceGraphData} 
                selectedMessage={selectedMessage}
                hoveredNodeId={hoveredNodeId}
              />
            </Box>
          </Box>
        ) : (
          <Paper sx={{ textAlign: 'center', p: 10, display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', height: '100%' }}>
            <Typography variant="h4" gutterBottom>
              No Chatbot Selected
            </Typography>
            <Typography variant="subtitle1">
              Please select a chatbot from the <a href="/chatbots">Chatbot Management</a> page.
            </Typography>
          </Paper>
        )}
      </Container>
    </Box>
  );
};

export default ChatPage;
