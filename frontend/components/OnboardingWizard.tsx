import React, { useState } from 'react';
import {
  Box,
  Stepper,
  Step,
  StepLabel,
  Button,
  Typography,
  Paper,
  Card,
  CardContent,
  CardActions,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  List,
  ListItem,
  ListItemIcon,
  ListItemText,
  Chip,
  useTheme,
  useMediaQuery
} from '@mui/material';
import {
  CheckCircle as CheckCircleIcon,
  Info as InfoIcon,
  Upload as UploadIcon,
  Chat as ChatIcon,
  AccountTree as GraphIcon,
  PlayArrow as PlayIcon,
  Close as CloseIcon
} from '@mui/icons-material';

interface OnboardingWizardProps {
  open: boolean;
  onClose: () => void;
  onComplete: () => void;
}

const steps = [
  'Welcome',
  'Understanding GraphRAG',
  'Create Your First Chatbot',
  'Upload Knowledge Sources',
  'Start Chatting'
];

const OnboardingWizard: React.FC<OnboardingWizardProps> = ({ open, onClose, onComplete }) => {
  const [activeStep, setActiveStep] = useState(0);
  const theme = useTheme();
  const isMobile = useMediaQuery(theme.breakpoints.down('sm'));

  const handleNext = () => {
    if (activeStep === steps.length - 1) {
      onComplete();
    } else {
      setActiveStep((prevActiveStep) => prevActiveStep + 1);
    }
  };

  const handleBack = () => {
    setActiveStep((prevActiveStep) => prevActiveStep - 1);
  };

  const handleSkip = () => {
    onComplete();
  };

  const renderStepContent = (step: number) => {
    switch (step) {
      case 0:
        return (
          <Box sx={{ textAlign: 'center', py: 4 }}>
            <GraphIcon sx={{ fontSize: 80, color: 'primary.main', mb: 2 }} />
            <Typography variant="h4" gutterBottom>
              Welcome to GraphRAG!
            </Typography>
            <Typography variant="body1" color="text.secondary" paragraph>
              GraphRAG combines the power of knowledge graphs with advanced AI to help you 
              chat with your documents in a more intelligent way.
            </Typography>
            <Typography variant="body1" color="text.secondary" paragraph>
              This quick tour will help you get started in just a few minutes.
            </Typography>
            <Box sx={{ mt: 4, display: 'flex', justifyContent: 'center', flexWrap: 'wrap', gap: 1 }}>
              <Chip icon={<InfoIcon />} label="Smart Document Understanding" />
              <Chip icon={<GraphIcon />} label="Knowledge Graphs" />
              <Chip icon={<ChatIcon />} label="Natural Conversations" />
            </Box>
          </Box>
        );

      case 1:
        return (
          <Box sx={{ py: 2 }}>
            <Typography variant="h5" gutterBottom>
              What is GraphRAG?
            </Typography>
            <Typography variant="body1" paragraph>
              GraphRAG (Graph-based Retrieval Augmented Generation) enhances traditional 
              document Q&A by creating knowledge graphs from your documents.
            </Typography>
            
            <Box sx={{ my: 3 }}>
              <Typography variant="h6" gutterBottom color="primary">
                Key Benefits:
              </Typography>
              <List>
                <ListItem>
                  <ListItemIcon>
                    <CheckCircleIcon color="success" />
                  </ListItemIcon>
                  <ListItemText
                    primary="Better Context Understanding"
                    secondary="Understands relationships between concepts in your documents"
                  />
                </ListItem>
                <ListItem>
                  <ListItemIcon>
                    <CheckCircleIcon color="success" />
                  </ListItemIcon>
                  <ListItemText
                    primary="More Accurate Answers"
                    secondary="Finds relevant information by following knowledge connections"
                  />
                </ListItem>
                <ListItem>
                  <ListItemIcon>
                    <CheckCircleIcon color="success" />
                  </ListItemIcon>
                  <ListItemText
                    primary="Visual Knowledge Exploration"
                    secondary="See how concepts are connected in interactive graphs"
                  />
                </ListItem>
              </List>
            </Box>

            <Paper sx={{ p: 2, bgcolor: 'info.light', color: 'info.contrastText' }}>
              <Typography variant="body2">
                <strong>Example:</strong> Instead of just finding documents that mention "revenue," 
                GraphRAG can understand the relationships between revenue, products, regions, 
                and time periods to give you more comprehensive answers.
              </Typography>
            </Paper>
          </Box>
        );

      case 2:
        return (
          <Box sx={{ py: 2 }}>
            <Typography variant="h5" gutterBottom>
              Create Your First Chatbot
            </Typography>
            <Typography variant="body1" paragraph>
              A chatbot in GraphRAG is a knowledge assistant trained on your specific documents.
              Each chatbot can be specialized for different purposes or document collections.
            </Typography>

            <Card sx={{ mb: 2, border: 1, borderColor: 'primary.main' }}>
              <CardContent>
                <Typography variant="h6" color="primary" gutterBottom>
                  💡 Pro Tip: Choose a Focused Topic
                </Typography>
                <Typography variant="body2">
                  For best results, create chatbots for specific domains like:
                </Typography>
                <Box sx={{ mt: 1, display: 'flex', flexWrap: 'wrap', gap: 0.5 }}>
                  <Chip size="small" label="Company Policies" />
                  <Chip size="small" label="Product Documentation" />
                  <Chip size="small" label="Research Papers" />
                  <Chip size="small" label="Technical Manuals" />
                </Box>
              </CardContent>
              <CardActions>
                <Button size="small" variant="outlined" startIcon={<PlayIcon />}>
                  I'll create one after this tour
                </Button>
              </CardActions>
            </Card>

            <Typography variant="body2" color="text.secondary">
              After this onboarding, you'll be taken to the chatbot creation page where you can:
            </Typography>
            <List dense>
              <ListItem sx={{ pl: 2 }}>
                <ListItemText 
                  primary="• Give your chatbot a descriptive name"
                  secondary="• Add a brief description of its purpose"
                />
              </ListItem>
            </List>
          </Box>
        );

      case 3:
        return (
          <Box sx={{ py: 2 }}>
            <Typography variant="h5" gutterBottom>
              Upload Knowledge Sources
            </Typography>
            <Typography variant="body1" paragraph>
              Your chatbot learns from the documents you upload. The system will process 
              these documents to build a knowledge graph.
            </Typography>

            <Box sx={{ my: 3 }}>
              <Typography variant="h6" gutterBottom>
                Supported File Types:
              </Typography>
              <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 1, mb: 2 }}>
                <Chip icon={<UploadIcon />} label="PDF Documents" />
                <Chip icon={<UploadIcon />} label="Word Documents" />
                <Chip icon={<UploadIcon />} label="Text Files" />
                <Chip icon={<UploadIcon />} label="Markdown Files" />
              </Box>
              
              <Typography variant="body2" color="text.secondary" gutterBottom>
                Maximum file size: 50MB per file
              </Typography>
            </Box>

            <Paper sx={{ p: 2, bgcolor: 'warning.light', color: 'warning.contrastText', mb: 2 }}>
              <Typography variant="body2">
                <strong>Processing Time:</strong> Depending on the size and number of documents, 
                it may take a few minutes to build the knowledge graph. You'll see a progress 
                indicator during this process.
              </Typography>
            </Paper>

            <Card variant="outlined">
              <CardContent>
                <Typography variant="h6" gutterBottom>
                  Best Practices:
                </Typography>
                <List dense>
                  <ListItem>
                    <ListItemText primary="Start with 5-10 high-quality documents" />
                  </ListItem>
                  <ListItem>
                    <ListItemText primary="Use documents that are text-heavy rather than image-heavy" />
                  </ListItem>
                  <ListItem>
                    <ListItemText primary="Upload related documents together for better connections" />
                  </ListItem>
                </List>
              </CardContent>
            </Card>
          </Box>
        );

      case 4:
        return (
          <Box sx={{ textAlign: 'center', py: 4 }}>
            <ChatIcon sx={{ fontSize: 80, color: 'success.main', mb: 2 }} />
            <Typography variant="h4" gutterBottom>
              You're All Set!
            </Typography>
            <Typography variant="body1" color="text.secondary" paragraph>
              Once your chatbot has processed your documents, you can start asking questions.
              The system will provide answers with source references and show you the 
              knowledge graph connections.
            </Typography>

            <Card sx={{ mt: 3, maxWidth: 500, mx: 'auto' }}>
              <CardContent>
                <Typography variant="h6" color="primary" gutterBottom>
                  Example Questions to Try:
                </Typography>
                <List dense>
                  <ListItem>
                    <ListItemText 
                      primary="What are the main topics covered in these documents?"
                      secondary="Great for getting an overview"
                    />
                  </ListItem>
                  <ListItem>
                    <ListItemText 
                      primary="How do [concept A] and [concept B] relate to each other?"
                      secondary="Leverages the knowledge graph connections"
                    />
                  </ListItem>
                  <ListItem>
                    <ListItemText 
                      primary="Can you summarize the key points about [specific topic]?"
                      secondary="Gets comprehensive information from multiple sources"
                    />
                  </ListItem>
                </List>
              </CardContent>
            </Card>

            <Typography variant="body2" color="text.secondary" sx={{ mt: 2 }}>
              Click "Get Started" to create your first chatbot!
            </Typography>
          </Box>
        );

      default:
        return <Typography>Unknown step</Typography>;
    }
  };

  return (
    <Dialog
      open={open}
      onClose={onClose}
      maxWidth="md"
      fullWidth
      fullScreen={isMobile}
      PaperProps={{
        sx: {
          minHeight: isMobile ? '100vh' : '70vh',
          maxHeight: isMobile ? '100vh' : '90vh'
        }
      }}
    >
      <DialogTitle sx={{ m: 0, p: 2, display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
        <Typography variant="h6">Getting Started with GraphRAG</Typography>
        <Button
          onClick={handleSkip}
          size="small"
          sx={{ minWidth: 'auto' }}
          aria-label="Skip onboarding"
        >
          <CloseIcon />
        </Button>
      </DialogTitle>

      <DialogContent sx={{ pb: 1 }}>
        <Stepper 
          activeStep={activeStep} 
          sx={{ mb: 4 }}
          orientation={isMobile ? 'vertical' : 'horizontal'}
        >
          {steps.map((label) => (
            <Step key={label}>
              <StepLabel>{label}</StepLabel>
            </Step>
          ))}
        </Stepper>

        <Box sx={{ minHeight: 300 }}>
          {renderStepContent(activeStep)}
        </Box>
      </DialogContent>

      <DialogActions sx={{ p: 2, pt: 1 }}>
        <Button
          onClick={handleSkip}
          sx={{ mr: 'auto' }}
          size={isMobile ? 'small' : 'medium'}
        >
          Skip Tour
        </Button>
        
        <Box sx={{ display: 'flex', gap: 1 }}>
          <Button
            disabled={activeStep === 0}
            onClick={handleBack}
            size={isMobile ? 'small' : 'medium'}
          >
            Back
          </Button>
          <Button
            variant="contained"
            onClick={handleNext}
            size={isMobile ? 'small' : 'medium'}
          >
            {activeStep === steps.length - 1 ? 'Get Started' : 'Next'}
          </Button>
        </Box>
      </DialogActions>
    </Dialog>
  );
};

export default OnboardingWizard;