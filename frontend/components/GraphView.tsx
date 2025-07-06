import React, { useEffect, useRef } from 'react';
import ForceGraph2D from 'react-force-graph-2d';
import { Box } from '@mui/material';

// Since @types/react-force-graph-2d is unavailable, we use a more generic type
interface GraphData {
  nodes: { id: any; [key: string]: any }[];
  links: { source: any; target: any; [key: string]: any }[];
}

interface GraphViewProps {
  graphData: GraphData | null;
}

const GraphView: React.FC<GraphViewProps> = ({ graphData }) => {
  const graphRef = useRef<any>(null); // Ref to access graph methods

  // Auto-zoom to fit the graph when data changes
  useEffect(() => {
    if (graphRef.current) {
      graphRef.current.zoomToFit(400); // Zoom to fit with 400ms duration
    }
  }, [graphData]);

  if (!graphData || !graphData.nodes || !graphData.links) {
    return <Box sx={{ display: 'flex', justifyContent: 'center', alignItems: 'center', height: '100%' }}>No graph data available.</Box>;
  }

  return (
    <Box sx={{ width: '100%', height: '100%' }}>
      <ForceGraph2D
        ref={graphRef}
        graphData={graphData}
        nodeLabel="label" // Display the 'label' property on hover
        nodeAutoColorBy="label" // Color nodes based on their label
        linkDirectionalArrowLength={3.5}
        linkDirectionalArrowRelPos={1}
        linkCurvature={0.1}
        cooldownTicks={100} // Stop the simulation sooner
        onEngineStop={() => graphRef.current?.zoomToFit(400)}
      />
    </Box>
  );
};

export default GraphView;
