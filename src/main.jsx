import { StrictMode } from 'react';
import { createRoot } from 'react-dom/client';
import './index.css';
import { HashRouter, Routes, Route } from 'react-router-dom';
import App from './App.jsx';
import AutoencoderPage from './Autoencoder.jsx';

createRoot(document.getElementById('root')).render(
  <StrictMode>
    <HashRouter>
      <Routes>
        <Route path="/" element={<App />} />
        <Route path="/encoder" element={<AutoencoderPage />} />
      </Routes>
    </HashRouter>
  </StrictMode>
);
