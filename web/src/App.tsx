import React, { useState, useEffect } from 'react';
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
  const [flask_response, set_flask_response] = useState({message: 'Connecting to API'} as IAPIResponse)
  const client = new APIClient()

  useEffect(() => {
    const getFlask = () => {
      if (flask_response.status === undefined) {
        client.getFlaskHello().then((data) => {
          set_flask_response(data)
        })
      }
    }
    const interval = setInterval(() => {
      getFlask();
    }, 1000);

    getFlask();
    return () => clearInterval(interval);
  }, [client, flask_response]);

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
      <Detector />
    </ThemeProvider>
  );
}

export default App;
