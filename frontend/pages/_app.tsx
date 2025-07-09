
import * as React from 'react';
import { AppProps } from 'next/app';
import { ThemeProvider, createTheme } from '@mui/material/styles';
import CssBaseline from '@mui/material/CssBaseline';
import Head from 'next/head';

const theme = createTheme({
  palette: {
    mode: 'dark',
    background: {
      default: '#121212', // A dark, but not pure black background
      paper: '#1e1e1e',   // A slightly lighter color for paper elements
    },
    primary: {
      main: '#90caf9', // A light blue for primary actions
    },
    secondary: {
      main: '#f48fb1', // A pink for secondary actions
    },
  },
});

export default function MyApp({ Component, pageProps }: AppProps) {
  return (
    <ThemeProvider theme={theme}>
      <Head>
        <meta name="viewport" content="initial-scale=1, width=device-width" />
      </Head>
      <CssBaseline />
      <Component {...pageProps} />
    </ThemeProvider>
  );
}
