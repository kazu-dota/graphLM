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
}

const GraphPanel: React.FC<GraphPanelProps> = ({ mainGraphData, isMainGraphLoading, referenceGraphData }) => {
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

  return (
    <Paper elevation={3} sx={{ height: '75vh', display: 'flex', flexDirection: 'column' }}>
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
              <GraphView graphData={mainGraphData} />
            )}
          </Box>
        )}
        {selectedTab === 1 && (
          <Box sx={{ height: '100%', width: '100%' }}>
            {referenceGraphData ? (
              <GraphView graphData={referenceGraphData} />
            ) : (
              <Box sx={{ display: 'flex', justifyContent: 'center', alignItems: 'center', height: '100%' }}>
                <Typography variant="body2">No reference graph available for the current chat.</Typography>
              </Box>
            )}
          </Box>
        )}
      </Box>
    </Paper>
  );
};

export default GraphPanel;
