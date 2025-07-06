
import React, { useCallback, useState } from 'react';
import { useDropzone } from 'react-dropzone';
import { Box, Typography, Paper, List, ListItem, ListItemText, IconButton, LinearProgress, Button } from '@mui/material';
import { uploadKnowledgeSource } from '../utils/api';
import CloudUploadIcon from '@mui/icons-material/CloudUpload';
import DeleteIcon from '@mui/icons-material/Delete';

interface KnowledgeSourceUploadProps {
  chatbotId: string;
}

const KnowledgeSourceUpload: React.FC<KnowledgeSourceUploadProps> = ({ chatbotId }) => {
  const [uploadedFiles, setUploadedFiles] = useState<File[]>([]);
  const [isUploading, setIsUploading] = useState(false);

  const onDrop = useCallback((acceptedFiles: File[]) => {
    setUploadedFiles(prevFiles => [...prevFiles, ...acceptedFiles]);
  }, []);

  const handleUpload = async () => {
    if (uploadedFiles.length === 0) {
      alert("Please select files to upload.");
      return;
    }
    setIsUploading(true);
    try {
      // This could be improved with parallel uploads and progress tracking for each file
      for (const file of uploadedFiles) {
        await uploadKnowledgeSource(chatbotId, file);
      }
      alert("Files uploaded successfully!");
      setUploadedFiles([]); // Clear the list after successful upload
    } catch (error) {
      alert("An error occurred during file upload.");
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
    </Box>
  );
};

export default KnowledgeSourceUpload;

