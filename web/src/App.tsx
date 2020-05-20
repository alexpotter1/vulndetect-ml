import React, { useEffect, useState } from 'react';
import { AppBar, Toolbar, IconButton, Typography, makeStyles, CssBaseline, createMuiTheme, ThemeProvider } from "@material-ui/core"
import Menu from "@material-ui/icons/Menu"
import APIClient, { IAPIResponse } from './api/APIClient';
import './App.css'
import { blue, red } from '@material-ui/core/colors';
import Detector from './Detector';

const theme = createMuiTheme({
  palette: {
    primary: blue,
    secondary: red
  },
});

const useStyles = makeStyles((theme) => ({
  root: {
    display: 'flex'
  },
  appBar: {
    zIndex: theme.zIndex.drawer + 1,
    transition: theme.transitions.create(['width', 'margin'], {
      easing: theme.transitions.easing.sharp,
      duration: theme.transitions.duration.leavingScreen,
    }),
  },
}));

function App() {
  const classes = useStyles();
  const [backendAlive, setBackendAlive] = useState(false);
  useEffect(() => {
    const backendHeartbeat = (): void => {
      APIClient.getBackendHeartbeat().then((data: IAPIResponse) => {
        if (data.status === '200 OK') {
          setBackendAlive(true);
        }
      }).catch((error) => {
        setBackendAlive(false);
      });
    }

    let heartbeatInterval = 10000;
    if (backendAlive) {
      heartbeatInterval = 20000;
    }

    const interval = setInterval(() => {
      backendHeartbeat();
    }, heartbeatInterval);

    backendHeartbeat();
    return () => clearInterval(interval);
  }, [backendAlive]);

  return (
    <ThemeProvider theme={theme}>
      <div className={classes.root}>
        <CssBaseline />
        <AppBar className={classes.appBar} position="static">
          <Toolbar>
            <IconButton edge="start" color="inherit" aria-label="menu">
              <Menu />
            </IconButton>
            <Typography variant="h6">
              Vulnerability Detector
            </Typography>
          </Toolbar>
        </AppBar>
      </div>
      <Detector backendAlive={backendAlive} />
    </ThemeProvider>
  );
}

export default App;
