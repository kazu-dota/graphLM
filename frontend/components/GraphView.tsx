import React, { useEffect, useRef, useState, useCallback } from 'react';
import ForceGraph2D from 'react-force-graph-2d';
import { Box, useTheme, Modal, Paper, Typography } from '@mui/material';

interface GraphData {
  nodes: { id: any; [key: string]: any }[];
  links: { source: any; target: any; [key: string]: any }[];
}

interface GraphViewProps {
  graphData: GraphData | null;
  highlightedNodes?: Set<string>;
  selectedMessage?: any | null;
}

// A more chic, modern color palette (based on Nord)
const NODE_COLORS = [
  '#bf616a', // nord8 (red)
  '#d08770', // nord9 (orange)
  '#ebcb8b', // nord10 (yellow)
  '#a3be8c', // nord11 (green)
  '#88c0d0', // nord14 (light blue)
  '#81a1c1', // nord15 (blue)
  '#b48ead', // nord12 (purple)
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


const GraphView: React.FC<GraphViewProps> = ({ graphData, highlightedNodes: propHighlightedNodes = new Set(), selectedMessage = null }) => {
  const theme = useTheme();
  const graphRef = useRef<any>(null);
  const [highlightNodes, setHighlightNodes] = useState(new Set());
  const [highlightLinks, setHighlightLinks] = useState(new Set());
  const [initialFitDone, setInitialFitDone] = useState(false);
  const [modalOpen, setModalOpen] = useState(false);
  const [selectedText, setSelectedText] = useState('');

  const handleOpenModal = (text: string) => {
    setSelectedText(text);
    setModalOpen(true);
  };

  const handleCloseModal = () => {
    setModalOpen(false);
    setSelectedText('');
  };

  useEffect(() => {
    // When graph data changes, reset the initial fit flag
    setInitialFitDone(false);
    setHighlightNodes(new Set());
    setHighlightLinks(new Set());
  }, [graphData]);

  const handleNodeClick = useCallback((node: any) => {
    // Check if the clicked node corresponds to a source document
    if (selectedMessage && selectedMessage.sources) {
      const source = selectedMessage.sources.find((s: any) => s.document_name === node.label);
      if (source && source.url) {
        window.open(`http://localhost:8000${source.url}`, '_blank');
        return; // Don't do the regular highlight if we opened a doc
      }
    }

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
  }, [graphRef, graphData, selectedMessage]);

  const handleLinkClick = useCallback((link: any) => {
    if (link.source_text) {
      handleOpenModal(link.source_text);
    }
  }, []);

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
            color = theme.palette.secondary.main; // Highlight color for hovered nodes
          }

          // Draw circle
          ctx.beginPath();
          ctx.arc(node.x!, node.y!, 5, 0, 2 * Math.PI, false);
          ctx.fillStyle = (isNodeHighlighted || highlightNodes.size === 0) ? color : 'rgba(100, 100, 100, 0.5)';
          ctx.fill();

          // --- Text rendering improvements ---
          const textX = node.x!;
          const textY = node.y! + 10;
          const shortLabel = label.length > 20 ? label.substring(0, 20) + '...' : label;

          ctx.font = `${fontSize}px Sans-Serif`;
          const textWidth = ctx.measureText(shortLabel).width;
          const bgDimensions = [textWidth, fontSize].map(n => n + fontSize * 0.2);

          // Draw background rectangle
          ctx.fillStyle = 'rgba(20, 20, 20, 0.8)';
          ctx.fillRect(textX - bgDimensions[0] / 2, textY - bgDimensions[1] / 2, bgDimensions[0], bgDimensions[1]);

          // Draw label text
          ctx.textAlign = 'center';
          ctx.textBaseline = 'middle';
          ctx.fillStyle = (isNodeHighlighted || highlightNodes.size === 0) ? 'rgba(255, 255, 255, 0.9)' : 'rgba(200, 200, 200, 0.8)';
          ctx.fillText(shortLabel, textX, textY);
        }}
        onNodeClick={handleNodeClick}
        onBackgroundClick={handleBackgroundClick}
        onLinkClick={handleLinkClick}

        // Link styling
        linkWidth={link => (highlightLinks.has(link) ? 2.5 : 1)}
        linkColor={link => (highlightLinks.size === 0 || highlightLinks.has(link)) ? '#aaa' : 'rgba(150, 150, 150, 0.3)'}
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
          ctx.fillStyle = 'rgba(20, 20, 20, 0.8)';
          ctx.fillRect(textPos.x - bgDimensions[0] / 2, textPos.y - bgDimensions[1] / 2, bgDimensions[0], bgDimensions[1]);
          
          // Draw text
          ctx.textAlign = 'center';
          ctx.textBaseline = 'middle';
          ctx.fillStyle = (highlightLinks.size === 0 || highlightLinks.has(link)) ? 'rgba(255, 255, 255, 0.9)' : 'rgba(200, 200, 200, 0.8)';
          ctx.fillText(label, textPos.x, textPos.y);
        }}
        
        cooldownTicks={100}
        onEngineStop={handleEngineStop}
      />
      <Modal
        open={modalOpen}
        onClose={handleCloseModal}
        aria-labelledby="source-text-modal-title"
        aria-describedby="source-text-modal-description"
      >
        <Paper sx={{ 
          position: 'absolute', 
          top: '50%', 
          left: '50%', 
          transform: 'translate(-50%, -50%)',
          width: 600,
          bgcolor: 'background.paper',
          border: '2px solid #000',
          boxShadow: 24,
          p: 4,
        }}>
          <Typography id="source-text-modal-title" variant="h6" component="h2">
            Source Text
          </Typography>
          <Typography id="source-text-modal-description" sx={{ mt: 2, whiteSpace: 'pre-wrap' }}>
            {selectedText}
          </Typography>
        </Paper>
      </Modal>
    </Box>
  );
};

export default GraphView;