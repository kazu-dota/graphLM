
import React, { useState } from 'react';
import { Box, Typography, TextField, Button, Stepper, Step, StepLabel, StepContent, Paper } from '@mui/material';
import { createChatbot } from '../utils/api';
import KnowledgeSourceUpload from './KnowledgeSourceUpload'; // Import the component

interface ChatbotCreationFormProps {
  onChatbotCreated: () => void;
}

const steps = ['Enter Chatbot Details', 'Upload Knowledge Sources', 'Finish'];

const ChatbotCreationForm: React.FC<ChatbotCreationFormProps> = ({ onChatbotCreated }) => {
  const [activeStep, setActiveStep] = useState(0);
  const [name, setName] = useState('');
  const [description, setDescription] = useState('');
  const [createdChatbotId, setCreatedChatbotId] = useState<string | null>(null);
  const [isCreating, setIsCreating] = useState(false);

  const handleNext = async () => {
    if (activeStep === 0) {
      if (!name) {
        alert('Please enter a chatbot name.');
        return;
      }
      setIsCreating(true);
      try {
        const chatbot = await createChatbot(name, description);
        setCreatedChatbotId(chatbot.id);
        onChatbotCreated(); // Refresh list in the background
        setActiveStep((prevActiveStep) => prevActiveStep + 1);
      } catch (error) {
        alert('Failed to create chatbot.');
      } finally {
        setIsCreating(false);
      }
    } else {
      setActiveStep((prevActiveStep) => prevActiveStep + 1);
    }
  };

  const handleBack = () => {
    setActiveStep((prevActiveStep) => prevActiveStep - 1);
  };

  const handleReset = () => {
    setActiveStep(0);
    setName('');
    setDescription('');
    setCreatedChatbotId(null);
  };

  return (
    <Box sx={{ width: '100%' }}>
      <Stepper activeStep={activeStep} orientation="vertical">
        {/* Step 1: Details */}
        <Step>
          <StepLabel>Enter Chatbot Details</StepLabel>
          <StepContent>
            <TextField
              label="Chatbot Name"
              variant="outlined"
              fullWidth
              value={name}
              onChange={(e) => setName(e.target.value)}
              required
              sx={{ mt: 1, mb: 2 }}
            />
            <TextField
              label="Description (Optional)"
              variant="outlined"
              fullWidth
              multiline
              rows={3}
              value={description}
              onChange={(e) => setDescription(e.target.value)}
              sx={{ mb: 2 }}
            />
            <Box sx={{ mb: 2 }}>
              <div>
                <Button
                  variant="contained"
                  onClick={handleNext}
                  disabled={isCreating || !name}
                  sx={{ mt: 1, mr: 1 }}
                >
                  {isCreating ? 'Creating...' : 'Continue'}
                </Button>
              </div>
            </Box>
          </StepContent>
        </Step>

        {/* Step 2: Upload */}
        <Step>
          <StepLabel>Upload Knowledge Sources</StepLabel>
          <StepContent>
            {createdChatbotId && <KnowledgeSourceUpload chatbotId={createdChatbotId} />}
            <Box sx={{ mb: 2 }}>
              <div>
                <Button
                  variant="contained"
                  onClick={handleNext}
                  sx={{ mt: 1, mr: 1 }}
                >
                  Continue
                </Button>
                <Button
                  onClick={handleBack}
                  sx={{ mt: 1, mr: 1 }}
                >
                  Back
                </Button>
              </div>
            </Box>
          </StepContent>
        </Step>
        
        {/* Step 3: Finish */}
        <Step>
            <StepLabel>Finish</StepLabel>
            <StepContent>
                <Typography>All steps completed - you&apos;re finished!</Typography>
                <Button onClick={handleReset} sx={{ mt: 1, mr: 1 }}>
                    Create Another Chatbot
                </Button>
            </StepContent>
        </Step>
      </Stepper>
    </Box>
  );
};

export default ChatbotCreationForm;
