import React, { useState, useEffect } from 'react';
import { Paper, Typography, Box, Tabs, Tab, CircularProgress } from '@mui/material';
import dynamic from 'next/dynamic';

const GraphView = dynamic(() => import('./GraphView'), {
  ssr: false,
  loading: () => <p>Loading graph...</p>
});

interface GraphData {
  nodes: any[];
  links: any[];
}

interface GraphPanelProps {
  mainGraphData: GraphData | null;
  isMainGraphLoading: boolean;
  referenceGraphData: GraphData | null;
  selectedMessage: any | null; // Receive the selected message
  hoveredNodeId: string | null; // New prop for hovered node
}

const GraphPanel: React.FC<GraphPanelProps> = ({ mainGraphData, isMainGraphLoading, referenceGraphData, selectedMessage, hoveredNodeId }) => {
  const [selectedTab, setSelectedTab] = useState(0);

  const handleChangeTab = (event: React.SyntheticEvent, newValue: number) => {
    setSelectedTab(newValue);
  };

  // Switch to reference graph tab if new reference data arrives
  useEffect(() => {
    if (referenceGraphData) {
      setSelectedTab(1);
    }
  }, [referenceGraphData]);

  // Determine highlighted nodes from the selected message's sources
  const highlightedNodes = new Set<string>();
  if (selectedTab === 0 && mainGraphData && selectedMessage?.sources) {
    // Highlight in main graph
    selectedMessage.sources.forEach((source: any) => {
      // This assumes the source object has a property that can be mapped to a node ID.
      // You might need to adjust this logic based on your actual data structure.
      // For example, if source.document_name is a node ID.
      if (source.document_name) {
        highlightedNodes.add(source.document_name);
      }
    });
  }
  // Note: Highlighting for reference graph (selectedTab === 1) could also be implemented here if needed.

  return (
    <Box sx={{ width: '100%', height: '100%' }}>
      <Paper elevation={3} sx={{ height: '100%', display: 'flex', flexDirection: 'column' }}>
        <Box sx={{ borderBottom: 1, borderColor: 'divider' }}>
          <Tabs value={selectedTab} onChange={handleChangeTab} aria-label="graph tabs">
            <Tab label="Knowledge Graph" />
            <Tab label="Reference Graph" disabled={!referenceGraphData} />
          </Tabs>
        </Box>
        <Box sx={{ flexGrow: 1, p: 2, position: 'relative' }}>
          {selectedTab === 0 && (
            <Box sx={{ height: '100%', width: '100%' }}>
              {isMainGraphLoading ? (
                <Box sx={{ display: 'flex', justifyContent: 'center', alignItems: 'center', height: '100%' }}>
                  <CircularProgress />
                </Box>
              ) : (
                <GraphView graphData={mainGraphData} highlightedNodes={hoveredNodeId ? new Set([hoveredNodeId]) : new Set()} selectedMessage={selectedMessage} />
              )}
            </Box>
          )}
          {selectedTab === 1 && (
            <Box sx={{ height: '100%', width: '100%' }}>
              {referenceGraphData ? (
                <GraphView graphData={referenceGraphData} selectedMessage={selectedMessage} />
              ) : (
                <Box sx={{ display: 'flex', justifyContent: 'center', alignItems: 'center', height: '100%' }}>
                  <Typography variant="body2">No reference graph available for the current chat.</Typography>
                </Box>
              )}
            </Box>
          )}
        </Box>
      </Paper>
    </Box>
  );
};

export default GraphPanel;
