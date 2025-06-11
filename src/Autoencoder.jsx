import { useState, useRef, useCallback, useEffect } from 'react';
import './App.css';
import { initializeAutoencoderModel, runMnistPrediction } from './services/mnistOnnxService';
import { Link } from 'react-router-dom';

function AutoencoderPage() {
  const canvasRef = useRef(null);
  const outputRef = useRef(null);
  const [session, setSession] = useState(null);
  const [modelStatus, setModelStatus] = useState('loading');
  const [modelError, setModelError] = useState(null);
  const [hasDrawn, setHasDrawn] = useState(false);
  const [isDrawing, setIsDrawing] = useState(false);
  const [lastPosition, setLastPosition] = useState(null);
  const [reconstructionHistory, setReconstructionHistory] = useState([]);  const [showHistory, setShowHistory] = useState(false);  const [showInfo, setShowInfo] = useState(false);
  const [autoReconstructTimer, setAutoReconstructTimer] = useState(null);
  const [selectedModel, setSelectedModel] = useState('autoencoder2.onnx'); // Default autoencoder model
  const [showInitialLoadMessage, setShowInitialLoadMessage] = useState(false);
  const isInitialLoadCycleRef = useRef(true); // Tracks if we are in the initial model load cycle
  // Initialize canvas for drawing
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    ctx.fillStyle = 'black';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';
    
    // Adjust line width based on screen size for better mobile experience
    const isMobile = window.innerWidth <= 768;
    const isSmallMobile = window.innerWidth <= 480;
    if (isSmallMobile) {
      ctx.lineWidth = 32; // Larger for very small screens
    } else if (isMobile) {
      ctx.lineWidth = 28; // Increased for mobile touch
    } else {
      ctx.lineWidth = 22; // Default for desktop
    }

    // Initialize output canvas
    const outputCanvas = outputRef.current;
    if (outputCanvas) {
      const outputCtx = outputCanvas.getContext('2d');
      outputCtx.fillStyle = 'black';
      outputCtx.fillRect(0, 0, outputCanvas.width, outputCanvas.height);
    }
  }, []);  // Load autoencoder model
  useEffect(() => {
    const loadModel = async () => {
      let currentLoadIsPartOfInitialCycle = false;
      if (isInitialLoadCycleRef.current) {
        setShowInitialLoadMessage(true);
        currentLoadIsPartOfInitialCycle = true;
      }
      setModelStatus('loading');
      setModelError(null); // Clear previous errors

      try {
        console.log(`Autoencoder: Attempting to initialize autoencoder model (${selectedModel})...`);
        const s = await initializeAutoencoderModel(selectedModel);
        setSession(s);
        if (s) {
          console.log('Autoencoder: Model loaded successfully');
          setModelStatus('loaded');
        } else {
          console.warn('Autoencoder: initializeAutoencoderModel returned null or undefined session');
          setModelError('Failed to initialize autoencoder session. Session is null.');
          setModelStatus('failed');
        }
      } catch (e) {
        console.error(`Autoencoder: Failed to load autoencoder model (${selectedModel})`, e);
        setModelError(`Failed to load model ${selectedModel}: ${e.message}`);
        setModelStatus('failed');
      } finally {
        if (currentLoadIsPartOfInitialCycle) {
          setShowInitialLoadMessage(false); // Hide the initial load message
          isInitialLoadCycleRef.current = false; // Mark the initial load cycle as complete
        }
      }
    };
    loadModel();
  }, [selectedModel]);

  // Auto-reconstruction helper function
  const startAutoReconstructTimer = useCallback(() => {
    // Clear any existing timer
    if (autoReconstructTimer) {
      clearTimeout(autoReconstructTimer);
    }
    
    // Only start timer if we have drawn content and model is loaded
    if (hasDrawn && session && modelStatus === 'loaded') {
      const timer = setTimeout(async () => {
        console.log('Auto-reconstruction triggered after 0.5s of inactivity');
        
        // Clear the timer reference
        setAutoReconstructTimer(null);
        
        // Run reconstruction logic directly
        setModelError(null);
        const imageVector = preprocessCanvas();
        const inputFloat32Array = Float32Array.from(imageVector);

        try {
          console.log("Autoencoder: Running auto-reconstruction with input tensor:", inputFloat32Array);
          const reconstructedOutput = await runMnistPrediction(session, inputFloat32Array);
          console.log("Autoencoder: Raw reconstruction output:", reconstructedOutput);

          if (!reconstructedOutput || reconstructedOutput.length !== 784) {
            console.error('Error during reconstruction or invalid output');
            setModelError('Auto-reconstruction failed. Invalid output from model.');
            return;
          }

          // Create reconstruction entry
          const reconstruction = {
            id: Date.now(),
            timestamp: new Date().toLocaleTimeString(),
            inputVector: imageVector,
            outputVector: Array.from(reconstructedOutput)
          };
          
          setReconstructionHistory(prev => [reconstruction, ...prev.slice(0, 9)]); // Keep last 10

          // Draw output on canvas
          const outputCanvas = outputRef.current;
          const outputCtx = outputCanvas.getContext('2d');
          const imgData = outputCtx.createImageData(28, 28);
          
          for (let i = 0; i < reconstructedOutput.length; i++) {
            const pixelValue = Math.floor(Math.max(0, Math.min(1, reconstructedOutput[i])) * 255);
            const pixelIndex = i * 4;
            imgData.data[pixelIndex] = pixelValue;     // R
            imgData.data[pixelIndex + 1] = pixelValue; // G
            imgData.data[pixelIndex + 2] = pixelValue; // B
            imgData.data[pixelIndex + 3] = 255;        // A
          }
          
          outputCtx.putImageData(imgData, 0, 0);

        } catch (err) {
          console.error('Autoencoder: Error during auto-reconstruction', err);
          setModelError(`Auto-reconstruction failed: ${err.message}`);
        }
      }, 500); // 0.5 seconds
      
      setAutoReconstructTimer(timer);
    }
  }, [autoReconstructTimer, hasDrawn, session, modelStatus]);

  // Cleanup timer on unmount
  useEffect(() => {
    return () => {
      if (autoReconstructTimer) {
        clearTimeout(autoReconstructTimer);
      }
    };
  }, [autoReconstructTimer]);
  // Drawing handlers
  const getCoordinates = (e) => {
    const canvas = canvasRef.current;
    if (!canvas) return { x: 0, y: 0 };
    const rect = canvas.getBoundingClientRect();
    const scaleX = canvas.width / rect.width;
    const scaleY = canvas.height / rect.height;
    
    if (e.touches && e.touches.length > 0) {
      // Touch event - use the first touch point
      const touch = e.touches[0];
      return {
        x: (touch.clientX - rect.left) * scaleX,
        y: (touch.clientY - rect.top) * scaleY
      };
    } else if (e.changedTouches && e.changedTouches.length > 0) {
      // Touch end event - use the first changed touch
      const touch = e.changedTouches[0];
      return {
        x: (touch.clientX - rect.left) * scaleX,
        y: (touch.clientY - rect.top) * scaleY
      };
    } else {
      // Mouse event
      return {
        x: (e.clientX - rect.left) * scaleX,
        y: (e.clientY - rect.top) * scaleY
      };
    }
  };

  // Helper to draw a single "fat" point with gradient
  const drawFatPoint = useCallback((x, y, ctx, pointLineWidth) => {
    const gradient = ctx.createRadialGradient(x, y, pointLineWidth / 7, x, y, pointLineWidth / 2);
    gradient.addColorStop(0, 'white'); // Center (most opaque)
    gradient.addColorStop(0.6, 'rgba(255, 255, 255, 0.7)'); // Middle
    gradient.addColorStop(1, 'rgba(255, 255, 255, 0)');   // Edge (transparent)

    ctx.fillStyle = gradient;
    ctx.beginPath();
    ctx.arc(x, y, pointLineWidth / 2, 0, 2 * Math.PI);
    ctx.fill();
  }, []);
  
  // Helper to draw a "fat" segment by interpolating points
  const drawFatSegment = useCallback((from, to, ctx, segmentLineWidth) => {
    const distance = Math.sqrt(Math.pow(to.x - from.x, 2) + Math.pow(to.y - from.y, 2));
    const steps = Math.max(1, Math.floor(distance / (segmentLineWidth / 4))); // Draw a point frequently

    for (let i = 0; i <= steps; i++) {
      const t = steps === 0 ? 0 : i / steps;
      const x = from.x + t * (to.x - from.x);
      const y = from.y + t * (to.y - from.y);
      drawFatPoint(x, y, ctx, segmentLineWidth);
    }
  }, [drawFatPoint]);
  const startDrawing = useCallback((e) => {
    e.preventDefault();
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    const currentLineWidth = parseFloat(ctx.lineWidth);

    setIsDrawing(true);
    setHasDrawn(true); // Mark that we have drawn content
    
    // Clear any existing auto-reconstruction timer when starting to draw
    if (autoReconstructTimer) {
      clearTimeout(autoReconstructTimer);
      setAutoReconstructTimer(null);
    }
    
    // Prevent scrolling on mobile while drawing
    document.body.style.overflow = 'hidden';
    document.documentElement.style.overflow = 'hidden';
    
    const coords = getCoordinates(e);
    setLastPosition(coords);
    drawFatPoint(coords.x, coords.y, ctx, currentLineWidth);
  }, [getCoordinates, drawFatPoint, autoReconstructTimer]);

  const draw = useCallback((e) => {
    e.preventDefault();
    if (!isDrawing || !lastPosition) return;

    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    const currentLineWidth = parseFloat(ctx.lineWidth);
    
    const currentCoords = getCoordinates(e);
    
    // Draw segment from last position to current
    drawFatSegment(lastPosition, currentCoords, ctx, currentLineWidth);
    
    // Draw an extra fat point at the current position to emphasize it and handle lingering
    drawFatPoint(currentCoords.x, currentCoords.y, ctx, currentLineWidth);

    setLastPosition(currentCoords);
  }, [isDrawing, lastPosition, getCoordinates, drawFatSegment, drawFatPoint]);
  const stopDrawing = useCallback((e) => {
    e.preventDefault();
    setIsDrawing(false);
    setLastPosition(null);
    
    // Re-enable scrolling on mobile
    document.body.style.overflow = '';
    document.documentElement.style.overflow = '';
    
    // Start auto-reconstruction timer when user stops drawing
    startAutoReconstructTimer();
  }, [startAutoReconstructTimer]);
  const clearCanvas = () => {
    const canvas = canvasRef.current;
    if (canvas) {
      const ctx = canvas.getContext('2d');
      ctx.fillStyle = 'black';
      ctx.fillRect(0, 0, canvas.width, canvas.height);
    }
    setHasDrawn(false); // Reset drawn content flag
    
    // Clear any existing auto-reconstruction timer
    if (autoReconstructTimer) {
      clearTimeout(autoReconstructTimer);
      setAutoReconstructTimer(null);
    }
    
    // Clear output canvas
    const outputCanvas = outputRef.current;
    if (outputCanvas) {
      const outputCtx = outputCanvas.getContext('2d');
      outputCtx.fillStyle = 'black';
      outputCtx.fillRect(0, 0, outputCanvas.width, outputCanvas.height);
    }
  };
  // Preprocessing function (same as App.jsx)
  const preprocessCanvas = () => {
    const sourceCanvas = canvasRef.current;
    
    const finalProcessedCanvas = document.createElement('canvas');
    const finalProcessedCtx = finalProcessedCanvas.getContext('2d');
    finalProcessedCanvas.width = 28;
    finalProcessedCanvas.height = 28;
    finalProcessedCtx.imageSmoothingEnabled = false;
    finalProcessedCtx.fillStyle = 'black'; // Ensure background of 28x28 canvas is black
    finalProcessedCtx.fillRect(0, 0, 28, 28);

    // 1. Get ImageData from the source 280x280 canvas
    const sourceCtx = sourceCanvas.getContext('2d');
    const sourceImageData = sourceCtx.getImageData(0, 0, sourceCanvas.width, sourceCanvas.height);
    const sourceData = sourceImageData.data;

    // 2. Find bounding box of the drawn content
    let minX = sourceCanvas.width;
    let minY = sourceCanvas.height;
    let maxX = -1;
    let maxY = -1;
    const threshold = 15; // Pixel intensity threshold (0-255) to be considered part of the digit

    for (let y = 0; y < sourceCanvas.height; y++) {
      for (let x = 0; x < sourceCanvas.width; x++) {
        const i = (y * sourceCanvas.width + x) * 4;
        const intensity = (sourceData[i] + sourceData[i+1] + sourceData[i+2]) / 3;
        const alpha = sourceData[i+3];

        if (intensity > threshold && alpha > 128) { // Check intensity and ensure it's not too transparent
          minX = Math.min(minX, x);
          minY = Math.min(minY, y);
          maxX = Math.max(maxX, x);
          maxY = Math.max(maxY, y);
        }
      }
    }

    // 3. If drawing found, process it; otherwise, finalProcessedCanvas remains black
    if (maxX >= minX && maxY >= minY) {
      const contentWidth = maxX - minX + 1;
      const contentHeight = maxY - minY + 1;

      // Create a temporary canvas for the cropped digit from source
      const cropCanvas = document.createElement('canvas');
      cropCanvas.width = contentWidth;
      cropCanvas.height = contentHeight;
      const cropCtx = cropCanvas.getContext('2d');
      // Draw the bounded part of the source canvas to the cropCanvas
      cropCtx.drawImage(sourceCanvas, minX, minY, contentWidth, contentHeight, 0, 0, contentWidth, contentHeight);

      // Define the target size for the digit within the 28x28 grid (e.g., 20x20 or 22x22)
      const targetDigitBoxSize = 20; // MNIST digits are often around this size in a 28x28 box
      const targetPadding = (28 - targetDigitBoxSize) / 2; // e.g., (28-20)/2 = 4 pixels padding

      // Scale the cropped image (cropCanvas) to fit into targetDigitBoxSize, preserving aspect ratio
      const scaleFactor = Math.min(targetDigitBoxSize / contentWidth, targetDigitBoxSize / contentHeight);
      const scaledWidth = contentWidth * scaleFactor;
      const scaledHeight = contentHeight * scaleFactor;

      // Calculate destination x, y on the 28x28 finalProcessedCanvas to center the scaled digit
      const destXOnFinal = targetPadding + (targetDigitBoxSize - scaledWidth) / 2;
      const destYOnFinal = targetPadding + (targetDigitBoxSize - scaledHeight) / 2;
      
      // Draw the scaled digit onto the 28x28 finalProcessedCanvas
      finalProcessedCtx.drawImage(cropCanvas, 0, 0, contentWidth, contentHeight, destXOnFinal, destYOnFinal, scaledWidth, scaledHeight);
    }
    
    const smallImageData = finalProcessedCtx.getImageData(0, 0, 28, 28);
    const data = smallImageData.data;
    
    const vector = [];

    for (let i = 0; i < data.length; i += 4) {
      const r = data[i];
      const g = data[i + 1];
      const b = data[i + 2];
      // Since drawing is white/transparent on black, (r+g+b)/3 is a good measure of intensity.
      const gray = (r + g + b) / 3; // gray is in the range 0-255
      
      const normalized = gray / 255.0; // Normalize to 0.0 - 1.0 range
      vector.push(normalized); // Push normalized value
    }
    // The vector will have 28*28 = 784 elements.
    return vector;
  };
  // Preprocess and run encoder
  const runEncode = async () => {
    if (!session || !hasDrawn) return;
    
    // Clear any existing auto-reconstruction timer when manually reconstructing
    if (autoReconstructTimer) {
      clearTimeout(autoReconstructTimer);
      setAutoReconstructTimer(null);
    }
    
    setModelError(null);
    
    try {
      console.log("Running autoencoder reconstruction...");
      
      // Get preprocessed input vector
      const imageVector = preprocessCanvas();
      const inputFloat32Array = Float32Array.from(imageVector);
      
      console.log("Input vector:", inputFloat32Array);
      
      // Run autoencoder inference
      const reconstructedOutput = await runMnistPrediction(session, inputFloat32Array);
      console.log("Autoencoder output:", reconstructedOutput);
      
      if (!reconstructedOutput || reconstructedOutput.length !== 784) {
        throw new Error('Invalid autoencoder output');
      }

      // Create reconstruction entry
      const reconstruction = {
        id: Date.now(),
        timestamp: new Date().toLocaleTimeString(),
        inputVector: imageVector,
        outputVector: Array.from(reconstructedOutput)
      };
      
      setReconstructionHistory(prev => [reconstruction, ...prev.slice(0, 9)]); // Keep last 10

      // Draw output on canvas
      const outputCanvas = outputRef.current;
      const outputCtx = outputCanvas.getContext('2d');
      const imgData = outputCtx.createImageData(28, 28);
      
      for (let i = 0; i < reconstructedOutput.length; i++) {
        const pixelValue = Math.floor(Math.max(0, Math.min(1, reconstructedOutput[i])) * 255);
        const pixelIndex = i * 4;
        imgData.data[pixelIndex] = pixelValue;     // R
        imgData.data[pixelIndex + 1] = pixelValue; // G
        imgData.data[pixelIndex + 2] = pixelValue; // B
        imgData.data[pixelIndex + 3] = 255;        // A
      }
      
      outputCtx.putImageData(imgData, 0, 0);
      
    } catch (error) {
      console.error('Autoencoder reconstruction failed:', error);
      setModelError(`Reconstruction failed: ${error.message}`);
    }
  };

  const clearHistory = () => {
    setReconstructionHistory([]);
  };
  return (
    <div className="app">
      <div className="container">
        <header className="header">
          <h1 className="title">
            <span className="gradient-text">MNIST</span> Autoencoder
          </h1>          <p className="subtitle">Draw a digit and see how the AI reconstructs it!</p>
          <div className="model-selector">
            <label htmlFor="autoencoder-select">Choose Autoencoder Model: </label>
            <select 
              id="autoencoder-select" 
              value={selectedModel} 
              onChange={(e) => setSelectedModel(e.target.value)}
              className="model-select"
            >
              <option value="autoencoder.onnx">testing - very bad!</option>
              <option value="autoencoder2.onnx">Advanced Autoencoder - Better quality</option>
            </select>
          </div>
          {showInitialLoadMessage && (
            <div className="initial-load-notification">
              Initializing autoencoder model for the first time. This may take up to 20 seconds. Please wait...
            </div>
          )}
          <div style={{ marginTop: '1rem' }}>
            <Link to="/" className="btn btn-secondary" style={{ textDecoration: 'none', display: 'inline-flex' }}>
              ← Back to Classifier
            </Link>
          </div>
        </header>

        <main className="main">
          <div className="canvas-section">
            <div className={`canvas-container ${isDrawing ? 'drawing' : ''}`}>
              <canvas
                ref={canvasRef}
                width={280}
                height={280}
                className="drawing-canvas"
                onMouseDown={startDrawing}
                onMouseMove={draw}
                onMouseUp={stopDrawing}
                onMouseLeave={stopDrawing}
                onTouchStart={startDrawing}
                onTouchMove={draw}
                onTouchEnd={stopDrawing}
              />
              <div className="canvas-overlay">
                <div className="instruction">Draw a digit (0-9)</div>
              </div>
            </div>
            
            <div className="controls">
              <button className="btn btn-primary" onClick={runEncode} disabled={!hasDrawn || modelStatus !== 'loaded'}>
                <span className="btn-icon">🔄</span>
                Reconstruct
              </button>
              <button className="btn btn-secondary" onClick={clearCanvas}>
                <span className="btn-icon">🗑️</span>
                Clear
              </button>              <div className={`model-status-indicator ${modelStatus}${autoReconstructTimer ? ' auto-predict-active' : ''}`}>
                <span className="status-dot"></span>
                <span className="status-text">
                  {autoReconstructTimer && 'Auto-reconstructing...'}
                  {!autoReconstructTimer && modelStatus === 'loading' && 'Loading Autoencoder'}
                  {!autoReconstructTimer && modelStatus === 'loaded' && 'Autoencoder Ready'}
                  {!autoReconstructTimer && modelStatus === 'failed' && 'Autoencoder Failed'}
                </span>
              </div>
            </div>

            {modelError && (
              <div style={{
                margin: '10px 0',
                padding: '10px',
                backgroundColor: '#f8d7da',
                border: '1px solid #f5c6cb',
                borderRadius: '4px',
                color: '#721c24',
                fontSize: '14px',
                textAlign: 'center',
                width: '100%',
                maxWidth: '400px'
              }}>
                <strong>Error:</strong> {modelError}
              </div>
            )}

            <div className="reconstruction-result">
              <div className="reconstruction-card">
                <h3>Reconstructed Output:</h3>
                <div className="output-canvas-container">
                  <canvas 
                    ref={outputRef} 
                    width={28} 
                    height={28} 
                    className="output-canvas"
                    style={{
                      border: '2px solid #e0e0e0',
                      borderRadius: '8px',
                      imageRendering: 'pixelated',
                      width: '140px',
                      height: '140px'
                    }}
                  />
                </div>
              </div>
            </div>
          </div>

          <aside className="sidebar">
            <div className="predictions-section">
              <div className="predictions-header">
                <button 
                  className="info-toggle"
                  onClick={() => setShowHistory(!showHistory)}
                  aria-expanded={showHistory}
                  style={{ width: 'auto', paddingRight: '30px', marginBottom: '1rem', display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexGrow: 1 }}
                >
                  <h3>Reconstruction History</h3>
                  <span className={`info-toggle-icon ${showHistory ? 'expanded' : ''}`}>
                    ▼
                  </span>
                </button>
                <div className="predictions-controls">
                  <button 
                    className="btn btn-small btn-danger" 
                    onClick={clearHistory}
                    disabled={reconstructionHistory.length === 0}
                    title="Clear history"
                  >
                    🗑️
                  </button>
                </div>
              </div>
              
              <div className={`predictions-list ${showHistory ? 'expanded' : ''}`} style={{ maxHeight: showHistory ? '400px' : '0', overflowY: 'auto', transition: 'max-height 0.3s ease' }}>
                {reconstructionHistory.length === 0 ? (
                  <div className="empty-state">
                    <div className="empty-icon">🎨</div>
                    <p>No reconstructions yet</p>
                    <p className="empty-subtitle">Draw and reconstruct to see history!</p>
                  </div>
                ) : (
                  reconstructionHistory.map((item) => (
                    <div key={item.id} className="prediction-item">
                      <div className="prediction-item-digit">📷</div>
                      <div className="prediction-item-info">
                        <div className="prediction-item-confidence">
                          Reconstruction #{item.id % 1000}
                        </div>
                        <div className="prediction-item-time">{item.timestamp}</div>
                      </div>
                    </div>
                  ))
                )}
              </div>
            </div>

            <div className="info-section">
              <button
                className="info-toggle"
                onClick={() => setShowInfo(!showInfo)}
                aria-expanded={showInfo}
              >
                How it works
                <span className={`info-toggle-icon ${showInfo ? 'expanded' : ''}`}>
                  ▼
                </span>
              </button>
              <div className={`info-content ${showInfo ? 'expanded' : ''}`}>
                <ul className="info-list">
                  <li>Draw a digit (0-9) on the canvas.</li>
                  <li>The image is preprocessed to 28×28 pixels.</li>
                  <li>The autoencoder compresses it to a smaller representation.</li>
                  <li>Then reconstructs it back to 28×28 pixels.</li>
                  <li>Compare the input and reconstructed output!</li>
                </ul>
              </div>
            </div>
          </aside>
        </main>
      </div>
    </div>
  );
}

export default AutoencoderPage;
