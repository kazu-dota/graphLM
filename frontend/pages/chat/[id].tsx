import React, { useState, useEffect } from 'react';
import ChatInterface from '../../components/ChatInterface';
import GraphPanel from '../../components/GraphPanel';
import dynamic from 'next/dynamic';
import { getGraphData } from '../../utils/api';
import { ChatbotStatus, Chatbot, GraphData, Message } from '../../types';
import { Box, AppBar, Toolbar, Typography, Container, Grid, useTheme, useMediaQuery, IconButton, Drawer } from '@mui/material';
import { Menu as MenuIcon, Close as CloseIcon } from '@mui/icons-material';
import { useRouter } from 'next/router';

const GraphView = dynamic(() => import('../../components/GraphView'), {
  ssr: false,
  loading: () => <p>Loading graph...</p>
});

// Remove duplicate interfaces - using types from common types file

const ChatPage: React.FC = () => {
  const router = useRouter();
  const { id: chatbotId } = router.query;
  const theme = useTheme();
  const isMobile = useMediaQuery(theme.breakpoints.down('md'));
  const isTablet = useMediaQuery(theme.breakpoints.down('lg'));

  const [selectedChatbotStatus, setSelectedChatbotStatus] = useState<ChatbotStatus>(ChatbotStatus.READY);
  const [mainGraphData, setMainGraphData] = useState<GraphData | null>(null);
  const [isGraphLoading, setIsGraphLoading] = useState<boolean>(false);
  const [referenceGraphData, setReferenceGraphData] = useState<GraphData | null>(null);
  const [hoveredNodeId, setHoveredNodeId] = useState<string | null>(null);
  const [selectedMessage, setSelectedMessage] = useState<Message | null>(null);
  const [mobileDrawerOpen, setMobileDrawerOpen] = useState<boolean>(false);

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

  const handleReferenceGraphData = (graphData: GraphData | null, message: Message | null) => {
    setReferenceGraphData(graphData);
    setSelectedMessage(message);
    // On mobile, automatically open drawer when graph data is available
    if (isMobile && graphData) {
      setMobileDrawerOpen(true);
    }
  };

  const toggleMobileDrawer = () => {
    setMobileDrawerOpen(!mobileDrawerOpen);
  };

  const GraphPanelComponent = () => (
    <GraphPanel 
      mainGraphData={mainGraphData} 
      isMainGraphLoading={isGraphLoading} 
      referenceGraphData={referenceGraphData} 
      selectedMessage={selectedMessage}
      hoveredNodeId={hoveredNodeId}
    />
  );

  return (
    <Box sx={{ flexGrow: 1, minHeight: '100vh', display: 'flex', flexDirection: 'column' }}>
      <AppBar position="static">
        <Toolbar>
          <Typography variant="h6" component="div" sx={{ flexGrow: 1 }}>
            GraphLM - Chat with Chatbot
          </Typography>
          {isMobile && (
            <IconButton
              color="inherit"
              edge="end"
              onClick={toggleMobileDrawer}
              aria-label="Open graph panel"
            >
              <MenuIcon />
            </IconButton>
          )}
        </Toolbar>
      </AppBar>

      <Container 
        maxWidth={false} 
        sx={{ 
          flex: 1,
          p: { xs: 1, sm: 2, md: 3 },
          display: 'flex',
          flexDirection: 'column'
        }}
      >
        {chatbotId ? (
          <>
            {/* Desktop/Tablet Layout */}
            {!isMobile && (
              <Grid 
                container 
                spacing={3} 
                sx={{ 
                  flex: 1,
                  height: { md: 'calc(100vh - 120px)', lg: 'calc(100vh - 100px)' },
                  minHeight: '600px'
                }}
              >
                <Grid 
                  item 
                  xs={12} 
                  md={7} 
                  lg={8}
                  sx={{ 
                    display: 'flex',
                    flexDirection: 'column',
                    minHeight: { xs: '400px', md: '500px' }
                  }}
                >
                  <Box 
                    sx={{ 
                      flex: 1,
                      border: 1, 
                      borderColor: 'divider', 
                      borderRadius: 1,
                      overflow: 'hidden',
                      display: 'flex',
                      flexDirection: 'column'
                    }}
                  >
                    <ChatInterface 
                      chatbotId={chatbotId as string} 
                      chatbotStatus={selectedChatbotStatus} 
                      onReferenceDataChange={handleReferenceGraphData}
                      onSourceHover={setHoveredNodeId}
                    />
                  </Box>
                </Grid>
                <Grid 
                  item 
                  xs={12} 
                  md={5} 
                  lg={4}
                  sx={{ 
                    display: 'flex',
                    flexDirection: 'column',
                    minHeight: { xs: '300px', md: '400px' }
                  }}
                >
                  <Box 
                    sx={{ 
                      flex: 1,
                      border: 1, 
                      borderColor: 'divider', 
                      borderRadius: 1,
                      overflow: 'hidden'
                    }}
                  >
                    <GraphPanelComponent />
                  </Box>
                </Grid>
              </Grid>
            )}

            {/* Mobile Layout */}
            {isMobile && (
              <>
                <Box 
                  sx={{ 
                    flex: 1,
                    border: 1, 
                    borderColor: 'divider', 
                    borderRadius: 1,
                    overflow: 'hidden',
                    minHeight: 'calc(100vh - 120px)',
                    display: 'flex',
                    flexDirection: 'column'
                  }}
                >
                  <ChatInterface 
                    chatbotId={chatbotId as string} 
                    chatbotStatus={selectedChatbotStatus} 
                    onReferenceDataChange={handleReferenceGraphData}
                    onSourceHover={setHoveredNodeId}
                  />
                </Box>

                {/* Mobile Drawer for Graph Panel */}
                <Drawer
                  anchor="bottom"
                  open={mobileDrawerOpen}
                  onClose={() => setMobileDrawerOpen(false)}
                  PaperProps={{
                    sx: {
                      height: '70vh',
                      borderTopLeftRadius: theme.spacing(2),
                      borderTopRightRadius: theme.spacing(2)
                    }
                  }}
                >
                  <Box sx={{ p: 1 }}>
                    <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 1 }}>
                      <Typography variant="h6">
                        Knowledge Graph
                      </Typography>
                      <IconButton onClick={() => setMobileDrawerOpen(false)}>
                        <CloseIcon />
                      </IconButton>
                    </Box>
                    <Box sx={{ height: 'calc(70vh - 80px)' }}>
                      <GraphPanelComponent />
                    </Box>
                  </Box>
                </Drawer>
              </>
            )}
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
