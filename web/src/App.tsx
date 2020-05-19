import React, { useState, useEffect } from 'react';
import logo from './logo.svg';
import './App.css';
import APIClient, { IAPIResponse } from './api/APIClient';

function App() {
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
    <div className="App">
      <header className="App-header">
        <img src={logo} className="App-logo" alt="logo" />
        <p>
          Edit <code>src/App.tsx</code> and save to reload.
        </p>
        <p>{flask_response.message}</p>
        <a
          className="App-link"
          href="https://reactjs.org"
          target="_blank"
          rel="noopener noreferrer"
        >
          Learn React
        </a>
      </header>
    </div>
  );
}

export default App;
