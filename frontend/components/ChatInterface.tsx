import React, { useState, useEffect, useRef, useCallback, useMemo } from 'react';
import { Box, Typography, TextField, Button, Paper, CircularProgress, List, ListItem, ListItemText, Divider, Grid, Card, CardContent, Link, ListItemButton } from '@mui/material';
import { chatWithBot } from '../utils/api';
import { ChatbotStatus, Message, Source, GraphData, StreamEvent, StreamData } from '../types';

interface ChatInterfaceProps {
  chatbotId: string;
  chatbotStatus: ChatbotStatus;
  onReferenceDataChange: (graphData: GraphData | null, message: Message | null) => void;
  onSourceHover: (nodeId: string | null) => void;
}

const renderMessageText = useMemo(() => (text: string, sources: Source[] = []) => {
  if (!text) return null;

  const parts: React.ReactNode[] = [];
  const citationRegex = /\(([^)]+\.(?:pdf|docx|txt|md))\)/g; // Matches (filename.ext)

  let lastIndex = 0;
  let match;

  while ((match = citationRegex.exec(text)) !== null) {
    const citationText = match[0]; // e.g., (document.pdf)
    const filename = match[1];    // e.g., document.pdf

    // Add text before the citation
    if (match.index > lastIndex) {
      parts.push(text.substring(lastIndex, match.index));
    }

    // Find the corresponding source
    const source = sources.find(s => s.filename === filename);

    if (source && source.url) {
      parts.push(
        <Link
          key={match.index}
          href={`http://localhost:8000${source.url}#page=${source.page_number || 1}&search=${encodeURIComponent(source.snippet || '')}`}
          target="_blank"
          rel="noopener noreferrer"
          sx={{ textDecoration: 'underline', color: 'inherit' }}
        >
          {citationText}
        </Link>
      );
    } else {
      // If source not found, render as plain text
      parts.push(citationText);
    }
    lastIndex = citationRegex.lastIndex;
  }

  // Add any remaining text after the last citation
  if (lastIndex < text.length) {
    parts.push(text.substring(lastIndex));
  }

  return <>{parts}</>;
}, []);

const ChatInterface: React.FC<ChatInterfaceProps> = ({ chatbotId, chatbotStatus, onReferenceDataChange, onSourceHover }) => {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState<string>('');
  const [loading, setLoading] = useState<boolean>(false);
  const [selectedMessage, setSelectedMessage] = useState<Message | null>(null);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  useEffect(() => {
    setMessages([]);
    setSelectedMessage(null);
    onReferenceDataChange(null, null);
  }, [chatbotId]);

  const handleSend = useCallback(async () => {
    if (!input.trim() || chatbotStatus !== ChatbotStatus.READY || loading) return;

    const userMessage: Message = {
      id: Date.now(),
      sender: 'user',
      text: input,
    };

    const botMessageId = Date.now() + 1;
    const botMessage: Message = {
      id: botMessageId,
      sender: 'bot',
      text: '', // Start with empty text
      sources: [],
      graphData: null,
      status: 'pending',
    };

    setMessages(prev => [...prev, userMessage, botMessage]);
    
    const currentInput = input;
    setInput('');
    setLoading(true);

    const handleStreamEvent = (event: any) => {
      if (event.error) {
        setMessages(prev => prev.map(msg => 
          msg.id === botMessageId 
            ? { ...msg, text: `Error: ${event.error}`, status: 'error' } 
            : msg
        ));
        setLoading(false);
        return;
      }

      switch (event.event) {
        case 'message':
          setMessages(prev => prev.map(msg => 
            msg.id === botMessageId 
              ? { ...msg, text: msg.text + event.data.text } 
              : msg
          ));
          break;
        case 'sources':
          setMessages(prev => prev.map(msg => 
            msg.id === botMessageId 
              ? { ...msg, sources: event.data.sources } 
              : msg
          ));
          break;
        case 'graph':
          setMessages(prev => prev.map(msg => {
            if (msg.id === botMessageId) {
              onReferenceDataChange(event.data, msg);
              return { ...msg, graphData: event.data };
            }
            return msg;
          }));
          break;
        case 'done':
          setMessages(prev => prev.map(msg => {
            if (msg.id === botMessageId) {
              const finalMessage = { ...msg, status: 'completed' as const };
              setSelectedMessage(finalMessage);
              onReferenceDataChange(finalMessage.graphData, finalMessage);
              return finalMessage;
            }
            return msg;
          }));
          setLoading(false);
          break;
      }
    };

    try {
      await chatWithBot(chatbotId, currentInput, handleStreamEvent);
    } catch (error: any) {
      const detail = error.toString() || 'Sorry, something went wrong.';
      setMessages(prev => prev.map(msg => 
        msg.id === botMessageId 
          ? { ...msg, text: `Error: ${detail}`, status: 'error' } 
          : msg
      ));
      setLoading(false);
    }
  }, [input, chatbotStatus, loading, chatbotId]);

  const handleSelectMessage = useCallback((message: Message) => {
    if (message.sender === 'bot' && message.status === 'completed') { // Only select completed bot messages
      setSelectedMessage(message);
      onReferenceDataChange(message.graphData, message);
    }
  }, [onReferenceDataChange]);

  const isChatDisabled = useMemo(() => chatbotStatus !== ChatbotStatus.READY, [chatbotStatus]);

  const renderedMessages = useMemo(() => 
    messages.map((msg) => (
      <Box 
        key={msg.id} 
        sx={{ mb: 2, cursor: msg.sender === 'bot' && msg.status === 'completed' ? 'pointer' : 'default' }} 
        onClick={() => handleSelectMessage(msg)}
        role={msg.sender === 'bot' && msg.status === 'completed' ? 'button' : undefined}
        tabIndex={msg.sender === 'bot' && msg.status === 'completed' ? 0 : undefined}
        onKeyDown={(e) => {
          if ((e.key === 'Enter' || e.key === ' ') && msg.sender === 'bot' && msg.status === 'completed') {
            handleSelectMessage(msg);
          }
        }}
        aria-label={msg.sender === 'bot' && msg.status === 'completed' ? `View details for bot message: ${msg.content?.substring(0, 100)}...` : undefined}
      >
        <Paper
          elevation={selectedMessage?.id === msg.id ? 4 : 1}
          sx={{
            p: 1.5,
            display: 'inline-block',
            minWidth: '10%',
            maxWidth: '90%',
            bgcolor: msg.sender === 'user' ? 'primary.dark' : 'background.paper',
            border: '1px solid',
            borderColor: selectedMessage?.id === msg.id ? 'primary.main' : 'divider',
            boxShadow: selectedMessage?.id === msg.id ? '0px 0px 8px rgba(0, 0, 0, 0.2)' : 'none',
            float: msg.sender === 'user' ? 'right' : 'left',
            clear: 'both',
          }}
          role="article"
          aria-label={`${msg.sender === 'user' ? 'User' : 'Bot'} message`}
        >
          <Typography variant="body1" sx={{ wordWrap: 'break-word' }}>
            {renderMessageText(msg.content, msg.sources)}
            {msg.sender === 'bot' && msg.status === 'pending' && (
              <CircularProgress size={16} sx={{ ml: 1 }} />
            )}
          </Typography>
          {msg.sender === 'bot' && msg.status === 'completed' && msg.sources && msg.sources.length > 0 && (
            <Box sx={{ mt: 2 }}>
              <Divider />
              <Typography variant="subtitle2" sx={{ mt: 1, mb: 1 }}>
                References:
              </Typography>
              <List dense>
                {msg.sources.filter(source => source.filename && source.url).map((source, i) => (
                  <Link href={`http://localhost:8000${source.url}#page=${source.page_number || 1}&search=${encodeURIComponent(source.snippet || '')}`} target="_blank" rel="noopener noreferrer" key={i} sx={{ textDecoration: 'none', color: 'inherit' }}>
                    <ListItemButton
                      onMouseEnter={() => onSourceHover(source.filename)}
                      onMouseLeave={() => onSourceHover(null)}
                    >
                      <ListItemText
                        primary={source.filename}
                        secondary={`Page: ${source.page_number || 'N/A'}`}
                      />
                    </ListItemButton>
                  </Link>
                ))}
              </List>
            </Box>
          )}
        </Paper>
      </Box>
    )), [messages, selectedMessage, handleSelectMessage, renderMessageText, onSourceHover]);

  return (
    <Box sx={{ width: '100%', height: '100%' }}>
      <Paper elevation={3} sx={{ height: '100%', display: 'flex', flexDirection: 'column' }}>
        <Typography variant="h6" sx={{ p: 2, borderBottom: 1, borderColor: 'divider' }}>
          Conversation
        </Typography>
        <Box sx={{ flexGrow: 1, overflowY: 'auto', p: 2 }}>
          {renderedMessages}
          <div ref={messagesEndRef} />
        </Box>
        <Divider />
        <Box sx={{ p: 2, display: 'flex', alignItems: 'center', gap: 1 }} role="form" aria-label="Chat input form">
          <TextField
            fullWidth
            variant="outlined"
            placeholder={isChatDisabled ? "Indexing is in progress..." : "Type your message..."}
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyPress={(e) => {
              if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                handleSend();
              }
            }}
            disabled={isChatDisabled || loading}
            multiline
            rows={3}
            aria-label="Type your message to the chatbot"
            aria-describedby="chat-input-help"
          />
          <Typography 
            id="chat-input-help" 
            variant="caption" 
            sx={{ display: 'none' }}
          >
            Press Enter to send your message, or Shift+Enter for a new line
          </Typography>
          <Box sx={{ position: 'relative' }}>
            <Button 
              variant="contained" 
              onClick={handleSend} 
              disabled={isChatDisabled || loading}
              aria-label={loading ? "Sending message..." : "Send message"}
            >
              Send
            </Button>
            {loading && (
              <CircularProgress
                size={24}
                sx={{
                  position: 'absolute',
                  top: '50%',
                  left: '50%',
                  marginTop: '-12px',
                  marginLeft: '-12px',
                }}
              />
            )}
          </Box>
        </Box>
      </Paper>
    </Box>
  );
};

export default React.memo(ChatInterface);