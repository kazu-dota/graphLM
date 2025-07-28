
import React, { useCallback, useState } from 'react';
import { useDropzone } from 'react-dropzone';
import { Box, Typography, Paper, List, ListItem, ListItemText, IconButton, LinearProgress, Button, ListItemIcon, Snackbar, Alert } from '@mui/material';
import { uploadKnowledgeSource } from '../utils/api';
import CloudUploadIcon from '@mui/icons-material/CloudUpload';
import DeleteIcon from '@mui/icons-material/Delete';
import InsertDriveFileIcon from '@mui/icons-material/InsertDriveFile';

interface KnowledgeSourceUploadProps {
  chatbotId: string;
}

interface SnackbarState {
  open: boolean;
  message: string;
  severity: 'success' | 'error' | 'warning' | 'info';
}

const KnowledgeSourceUpload: React.FC<KnowledgeSourceUploadProps> = ({ chatbotId }) => {
  const [uploadedFiles, setUploadedFiles] = useState<File[]>([]);
  const [isUploading, setIsUploading] = useState(false);
  const [snackbar, setSnackbar] = useState<SnackbarState>({
    open: false,
    message: '',
    severity: 'info'
  });

  const onDrop = useCallback((acceptedFiles: File[]) => {
    setUploadedFiles(prevFiles => [...prevFiles, ...acceptedFiles]);
  }, []);

  const showSnackbar = (message: string, severity: SnackbarState['severity']) => {
    setSnackbar({ open: true, message, severity });
  };

  const handleUpload = async () => {
    if (uploadedFiles.length === 0) {
      showSnackbar("Please select files to upload before proceeding.", 'warning');
      return;
    }
    
    setIsUploading(true);
    let successCount = 0;
    let failedFiles: string[] = [];
    
    try {
      for (const file of uploadedFiles) {
        try {
          await uploadKnowledgeSource(chatbotId, file);
          successCount++;
        } catch (error: any) {
          failedFiles.push(file.name);
          console.error(`Failed to upload ${file.name}:`, error);
        }
      }
      
      if (successCount === uploadedFiles.length) {
        showSnackbar(`Successfully uploaded ${successCount} file${successCount > 1 ? 's' : ''}!`, 'success');
        setUploadedFiles([]);
      } else if (successCount > 0) {
        showSnackbar(`Uploaded ${successCount} file${successCount > 1 ? 's' : ''}, but ${failedFiles.length} failed: ${failedFiles.join(', ')}`, 'warning');
      } else {
        showSnackbar(`Failed to upload files: ${failedFiles.join(', ')}. Please check file format and size.`, 'error');
      }
    } catch (error: any) {
      console.error('Upload error:', error);
      showSnackbar("An unexpected error occurred during upload. Please try again.", 'error');
    } finally {
      setIsUploading(false);
    }
  };

  const removeFile = (fileToRemove: File) => {
    setUploadedFiles(prevFiles => prevFiles.filter(file => file !== fileToRemove));
  };

  const { getRootProps, getInputProps, isDragActive } = useDropzone({ onDrop });

  return (
    <Box>
      <Paper
        {...getRootProps()}
        sx={{
          p: 4,
          border: '2px dashed grey',
          borderColor: isDragActive ? 'primary.main' : 'grey.500',
          textAlign: 'center',
          cursor: 'pointer',
          mb: 2
        }}
      >
        <input {...getInputProps()} />
        <CloudUploadIcon sx={{ fontSize: 48, color: 'grey.500' }} />
        <Typography>Drag & drop some files here, or click to select files</Typography>
      </Paper>
      {uploadedFiles.length > 0 && (
        <Box>
          <Typography variant="h6">Files to upload:</Typography>
          <List>
            {uploadedFiles.map((file, index) => (
              <ListItem key={index} secondaryAction={<IconButton edge="end" aria-label="delete" onClick={() => removeFile(file)}><DeleteIcon /></IconButton>}>
                <ListItemIcon>
                  <InsertDriveFileIcon />
                </ListItemIcon>
                <ListItemText primary={file.name} secondary={`${(file.size / 1024).toFixed(2)} KB`} />
              </ListItem>
            ))}
          </List>
          <Button variant="contained" onClick={handleUpload} disabled={isUploading} sx={{ mt: 2 }}>
            {isUploading ? 'Uploading...' : 'Upload Files'}
          </Button>
          {isUploading && <LinearProgress sx={{ mt: 1 }} />}
        </Box>
      )}
      
      <Snackbar
        open={snackbar.open}
        autoHideDuration={6000}
        onClose={() => setSnackbar(prev => ({ ...prev, open: false }))}
      >
        <Alert 
          onClose={() => setSnackbar(prev => ({ ...prev, open: false }))} 
          severity={snackbar.severity}
          sx={{ width: '100%' }}
        >
          {snackbar.message}
        </Alert>
      </Snackbar>
    </Box>
  );
};

export default KnowledgeSourceUpload;

