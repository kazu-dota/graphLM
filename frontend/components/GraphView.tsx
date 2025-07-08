import React, { useEffect, useRef, useState, useCallback } from 'react';
import ForceGraph2D from 'react-force-graph-2d';
import { Box } from '@mui/material';

interface GraphData {
  nodes: { id: any; [key: string]: any }[];
  links: { source: any; target: any; [key: string]: any }[];
}

interface GraphViewProps {
  graphData: GraphData | null;
  highlightedNodes?: Set<string>;
}

// Predefined color palette for node categories
const NODE_COLORS = [
  '#1f77b4', // muted blue
  '#ff7f0e', // safety orange
  '#2ca02c', // cooked asparagus green
  '#d62728', // brick red
  '#9467bd', // muted purple
  '#8c564b', // chestnut brown
  '#e377c2', // raspberry yogurt pink
  '#7f7f7f', // middle gray
  '#bcbd22', // curry yellow-green
  '#17becf'  // blue-teal
];

// Helper to get a consistent color based on node label (category)
const getNodeColor = (label: string) => {
  if (!label) return '#a0a0a0';
  // Simple hash to pick a color from the palette
  let hash = 0;
  for (let i = 0; i < label.length; i++) {
    hash = label.charCodeAt(i) + ((hash << 5) - hash);
  }
  return NODE_COLORS[Math.abs(hash) % NODE_COLORS.length];
};


const GraphView: React.FC<GraphViewProps> = ({ graphData, highlightedNodes: propHighlightedNodes = new Set() }) => {
  const graphRef = useRef<any>(null);
  const [highlightNodes, setHighlightNodes] = useState(new Set());
  const [highlightLinks, setHighlightLinks] = useState(new Set());
  const [initialFitDone, setInitialFitDone] = useState(false);

  useEffect(() => {
    // When graph data changes, reset the initial fit flag
    setInitialFitDone(false);
    setHighlightNodes(new Set());
    setHighlightLinks(new Set());
  }, [graphData]);

  const handleNodeClick = useCallback((node: any) => {
    // Center the view on the clicked node and highlight
    if (graphRef.current) {
      graphRef.current.centerAt(node.x, node.y, 1000);
      graphRef.current.zoom(2.5, 500);
    }

    const newHighlightNodes = new Set();
    const newHighlightLinks = new Set();
    if (node) {
        newHighlightNodes.add(node);
        graphData?.links.forEach(link => {
            if (link.source === node || link.target === node) {
                newHighlightLinks.add(link);
                newHighlightNodes.add(link.source);
                newHighlightNodes.add(link.target);
            }
        });
    }
    setHighlightNodes(newHighlightNodes);
    setHighlightLinks(newHighlightLinks);
  }, [graphRef, graphData]);

  const handleBackgroundClick = useCallback(() => {
    // Reset highlights
    setHighlightNodes(new Set());
    setHighlightLinks(new Set());
    if (graphRef.current) {
        graphRef.current.zoomToFit(400, 500);
    }
  }, [graphRef]);

  const handleEngineStop = useCallback(() => {
    if (graphRef.current && !initialFitDone) {
      graphRef.current.zoomToFit(400, 500);
      setInitialFitDone(true);
    }
  }, [initialFitDone]);


  if (!graphData || !graphData.nodes || !graphData.links) {
    return <Box sx={{ display: 'flex', justifyContent: 'center', alignItems: 'center', height: '100%' }}>No graph data available.</Box>;
  }

  return (
    <Box sx={{ width: '100%', height: '100%', position: 'relative' }}>
      <ForceGraph2D
        ref={graphRef}
        graphData={graphData}
        nodeLabel={node => `<div><b>${node.label}</b></div>`}
        // Physics engine configuration for layout optimization
        d3ForceManyBody={-300} // Increased repulsion strength
        d3ForceLink={100} // Increased link distance
        d3AlphaDecay={0.02} // Slower decay for more stable layout
        d3VelocityDecay={0.4} // Slower velocity decay

        // Node styling
        nodeCanvasObject={(node, ctx, globalScale) => {
          const label = node.label || node.id;
          const fontSize = 14 / globalScale; 
          const isNodeHighlighted = highlightNodes.has(node);
          const isPropHighlighted = propHighlightedNodes.has(node.id);

          // Use getNodeColor for consistent coloring
          let color = getNodeColor(node.label || '');
          if (isPropHighlighted) {
            color = 'red';
          }

          // Draw circle
          ctx.beginPath();
          ctx.arc(node.x!, node.y!, 5, 0, 2 * Math.PI, false);
          ctx.fillStyle = (isNodeHighlighted || highlightNodes.size === 0) ? color : 'rgba(200, 200, 200, 0.75)';
          ctx.fill();

          // --- Text rendering improvements ---
          const textX = node.x!;
          const textY = node.y! + 10; 
          const shortLabel = label.length > 20 ? label.substring(0, 20) + '...' : label;
          
          ctx.font = `${fontSize}px Sans-Serif`;
          const textWidth = ctx.measureText(shortLabel).width;
          const bgDimensions = [textWidth, fontSize].map(n => n + fontSize * 0.2); 

          // Draw background rectangle
          ctx.fillStyle = 'rgba(255, 255, 255, 0.8)';
          ctx.fillRect(textX - bgDimensions[0] / 2, textY - bgDimensions[1] / 2, bgDimensions[0], bgDimensions[1]);

          // Draw label text
          ctx.textAlign = 'center';
          ctx.textBaseline = 'middle';
          ctx.fillStyle = (isNodeHighlighted || highlightNodes.size === 0) ? 'black' : 'rgba(0, 0, 0, 0.7)';
          ctx.fillText(shortLabel, textX, textY);
        }}
        onNodeClick={handleNodeClick}
        onBackgroundClick={handleBackgroundClick}

        // Link styling
        linkWidth={link => (highlightLinks.has(link) ? 2.5 : 1)}
        linkColor={link => (highlightLinks.size === 0 || highlightLinks.has(link)) ? '#666' : 'rgba(150, 150, 150, 0.3)'}
        linkDirectionalArrowLength={3.5}
        linkDirectionalArrowRelPos={1}
        linkCurvature={0.1}
        
        // Add link label
        linkCanvasObjectMode={() => 'after'}
        linkCanvasObject={(link, ctx, globalScale) => {
          const label = link.label || '';
          if (!label || globalScale < 1.2) return; 

          const start = link.source;
          const end = link.target;

          if (typeof start !== 'object' || typeof end !== 'object') return;

          const textPos = {
            x: start.x! + (end.x! - start.x!) / 2,
            y: start.y! + (end.y! - start.y!) / 2
          };

          const fontSize = 10 / globalScale; 
          ctx.font = `${fontSize}px Sans-Serif`;
          const textWidth = ctx.measureText(label).width;
          const bgDimensions = [textWidth, fontSize].map(n => n + fontSize * 0.2);

          // Draw background
          ctx.fillStyle = 'rgba(255, 255, 255, 0.8)';
          ctx.fillRect(textPos.x - bgDimensions[0] / 2, textPos.y - bgDimensions[1] / 2, bgDimensions[0], bgDimensions[1]);
          
          // Draw text
          ctx.textAlign = 'center';
          ctx.textBaseline = 'middle';
          ctx.fillStyle = (highlightLinks.size === 0 || highlightLinks.has(link)) ? 'black' : 'rgba(0, 0, 0, 0.5)';
          ctx.fillText(label, textPos.x, textPos.y);
        }}
        
        cooldownTicks={100}
        onEngineStop={handleEngineStop}
      />
    </Box>
  );
};

export default GraphView;