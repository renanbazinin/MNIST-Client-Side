import { useState, useRef, useCallback, useEffect } from 'react';
import './App.css';
import * as ort from 'onnxruntime-web'; // Kept as ONNX runtime is fundamental
import { initializeMnistModel, runMnistPrediction } from './services/mnistOnnxService';

function App() {
  const canvasRef = useRef(null);
  const [isDrawing, setIsDrawing] = useState(false);
  const [lastPosition, setLastPosition] = useState(null); // Added for new drawing logic
  const [predictions, setPredictions] = useState([]);
  const [currentPrediction, setCurrentPrediction] = useState(null);
  const [showInfo, setShowInfo] = useState(false); // Already defaults to collapsed
  const [showPredictions, setShowPredictions] = useState(false); // New state for predictions visibility
  const [showLogs, setShowLogs] = useState(false); // State for logs visibility
  const [top4Predictions, setTop4Predictions] = useState([]); // State for top 4 predictions
  const [session, setSession] = useState(null);
  const [modelError, setModelError] = useState(null); // For displaying model loading errors

  // Initialize canvas
  useEffect(() => {
    const canvas = canvasRef.current;
    if (canvas) {
      const ctx = canvas.getContext('2d');
      ctx.fillStyle = 'black';
      ctx.fillRect(0, 0, canvas.width, canvas.height);
      ctx.lineCap = 'round'; // Good for connecting segments if we were stroking
      ctx.lineJoin = 'round'; // Good for connecting segments if we were stroking
      // ctx.strokeStyle = 'white'; // No longer using a simple strokeStyle
      
      const isMobile = window.innerWidth <= 768;
      ctx.lineWidth = isMobile ? 28 : 22; // Increased for "fat finger" effect
    }
  }, []);

  // Load ONNX model
  useEffect(() => {
    const loadModel = async () => {
      try {
        setModelError(null); // Clear previous errors
        console.log("App.jsx: Attempting to initialize MNIST ONNX model...");
        const newSession = await initializeMnistModel();
        setSession(newSession);
        if (newSession) {
          console.debug('App.jsx: MNIST ONNX session inputs:', newSession.inputNames);
          console.debug('App.jsx: MNIST ONNX session outputs:', newSession.outputNames);
          console.log("App.jsx: MNIST ONNX model session initialized and set in state.");
        } else {
          console.warn('App.jsx: initializeMnistModel returned null or undefined session');
          setModelError('Failed to initialize model session. Session is null.');
        }
      } catch (err) {
        console.error('App.jsx: Failed to load MNIST ONNX model', err);
        setModelError(`Failed to load model: ${err.message}`);
      }
    };
    loadModel();
  }, []);

  const getCoordinates = (e) => {
    const canvas = canvasRef.current;
    if (!canvas) return { x: 0, y: 0 }; // Guard against null canvas
    const rect = canvas.getBoundingClientRect();
    const scaleX = canvas.width / rect.width;
    const scaleY = canvas.height / rect.height;
    
    if (e.touches) {
      // Touch event
      return {
        x: (e.touches[0].clientX - rect.left) * scaleX,
        y: (e.touches[0].clientY - rect.top) * scaleY
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
    const coords = getCoordinates(e);
    setLastPosition(coords);
    drawFatPoint(coords.x, coords.y, ctx, currentLineWidth); // Draw initial point
  }, [getCoordinates, drawFatPoint]);

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
    setLastPosition(null); // Reset last position
  }, []);

  const clearCanvas = () => {
    const canvas = canvasRef.current;
    if (canvas) {
      const ctx = canvas.getContext('2d');
      ctx.fillStyle = 'black';
      ctx.fillRect(0, 0, canvas.width, canvas.height);
    }
    setCurrentPrediction(null);
  };

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
        // We draw with white/transparent gradients on a black background.
        // The R, G, B values will be similar for grayscale parts of the drawing.
        // Alpha (sourceData[i+3]) is also important. If alpha is 0, it's transparent.
        // A simple intensity check on RGB is fine if alpha is high enough.
        // Consider (r+g+b)/3 > threshold AND alpha > threshold_alpha if needed.
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
      // Alpha is data[i+3]
      // Since drawing is white/transparent on black, (r+g+b)/3 is a good measure of intensity.
      const gray = (r + g + b) / 3; // gray is in the range 0-255
      
      const normalized = gray / 255.0; // Normalize to 0.0 - 1.0 range
      vector.push(normalized); // Push normalized value
    }
    // The vector will have 28*28 = 784 elements.
    return vector;
  };

  const predict = useCallback(async () => {
    if (!session) {
      console.warn('App.jsx: ONNX session not loaded yet');
      setModelError('Model not loaded. Please wait or try refreshing.');
      return;
    }
    setModelError(null);

    const imageVector = preprocessCanvas();
    const inputFloat32Array = Float32Array.from(imageVector);

    try {
      console.log("App.jsx: Running prediction with input tensor:", inputFloat32Array);
      const outputProbabilities = await runMnistPrediction(session, inputFloat32Array);
      console.log("App.jsx: Raw ONNX softmax output:", outputProbabilities); // Log softmax output

      if (!outputProbabilities || outputProbabilities.length === 0) {
        console.error('Error during ONNX inference or no data returned');
        setModelError('Prediction failed. No output from model.');
        setTop4Predictions([]); // Clear top 4 predictions on error
        return;
      }

      const probs = Array.from(outputProbabilities);
      const indexedProbs = probs.map((prob, index) => ({ digit: index, probability: prob }));
      indexedProbs.sort((a, b) => b.probability - a.probability); // Sort by probability descending
      setTop4Predictions(indexedProbs.slice(0, 4)); // Store top 4

      const maxProb = indexedProbs[0].probability;
      const predIndex = indexedProbs[0].digit;

      const newPrediction = {
        digit: predIndex,
        confidence: maxProb, // Store as number, format in JSX
        timestamp: new Date().toLocaleTimeString(),
        id: Date.now()
      };

      setCurrentPrediction(newPrediction);
      setPredictions(prevPredictions => [newPrediction, ...prevPredictions]);

    } catch (err) {
      console.error('App.jsx: Error during prediction', err);
      setModelError(`Prediction failed: ${err.message}`);
      setTop4Predictions([]); // Clear top 4 predictions on error
    }
  }, [session]); // Removed preprocessCanvas from dependencies as it's called within

  const undoLastPrediction = () => {
    setPredictions(prev => prev.slice(1));
    if (predictions.length <= 1) {
      setCurrentPrediction(null);
    } else {
      setCurrentPrediction(predictions[1]); // predictions[0] is the one being removed, so predictions[1] becomes the new current.
    }
  };

  const copyToClipboard = () => {
    const text = predictions.map(p => `${p.digit} (Confidence: ${(p.confidence * 100).toFixed(1)}%)`).join('\\n');
    navigator.clipboard.writeText(text);
  };

  const resetPredictions = () => {
    setPredictions([]);
    setCurrentPrediction(null);
    setTop4Predictions([]); // Clear top 4 predictions on reset
  };

  return (
    <div className="app">
      <div className="container">
        <header className="header">
          <h1 className="title">
            <span className="gradient-text">MNIST</span> Digit Recognition
          </h1>
          <p className="subtitle">Draw a digit and let AI predict what you wrote!</p>
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
                onMouseLeave={stopDrawing} // Added to stop drawing if mouse leaves canvas
                onTouchStart={startDrawing}
                onTouchMove={draw}
                onTouchEnd={stopDrawing}
              />
              <div className="canvas-overlay">
                <div className="instruction">Draw a digit (0-9)</div>
              </div>
            </div>
            
            <div className="controls">
              <button className="btn btn-primary" onClick={predict}>
                <span className="btn-icon">🎯</span>
                Predict
              </button>
              <button className="btn btn-secondary" onClick={clearCanvas}>
                <span className="btn-icon">🗑️</span>
                Clear
              </button>
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

            {currentPrediction && (
              <div className="prediction-result">
                <div className="prediction-card">
                  <div className="prediction-digit">{currentPrediction.digit}</div>
                  <div className="prediction-info">
                    <div className="confidence">
                      Confidence: <span className="confidence-value">{(currentPrediction.confidence * 100).toFixed(1)}%</span>
                    </div>
                    <div className="timestamp">{currentPrediction.timestamp}</div>
                  </div>
                </div>
              </div>
            )}
          </div>

          <aside className="sidebar">
            <div className="predictions-section">
              <div className="predictions-header">
                <button 
                  className="info-toggle" // Re-using info-toggle style for consistency
                  onClick={() => setShowPredictions(!showPredictions)}
                  aria-expanded={showPredictions}
                  style={{ width: 'auto', paddingRight: '30px', marginBottom: '1rem', display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexGrow: 1 }} // Added style for button
                >
                  <h3>Recent Predictions</h3>
                  <span className={`info-toggle-icon ${showPredictions ? 'expanded' : ''}`}>
                    ▼
                  </span>
                </button>
                <div className="predictions-controls">
                  <button 
                    className="btn btn-small" 
                    onClick={undoLastPrediction}
                    disabled={predictions.length === 0}
                    title="Undo last prediction"
                  >
                    ↶
                  </button>
                  <button 
                    className="btn btn-small" 
                    onClick={copyToClipboard}
                    disabled={predictions.length === 0}
                    title="Copy to clipboard"
                  >
                    📋
                  </button>
                  <button 
                    className="btn btn-small btn-danger" 
                    onClick={resetPredictions}
                    disabled={predictions.length === 0}
                    title="Clear all predictions"
                  >
                    🗑️
                  </button>
                </div>
              </div>
              
              <div className={`predictions-list ${showPredictions ? 'expanded' : ''}`} style={{ maxHeight: showPredictions ? '400px' : '0', overflowY: 'auto', transition: 'max-height 0.3s ease' }}>
                {predictions.length === 0 ? (
                  <div className="empty-state">
                    <div className="empty-icon">✨</div>
                    <p>No predictions yet</p>
                    <p className="empty-subtitle">Draw a digit to get started!</p>
                  </div>
                ) : (
                  predictions.map((prediction) => (
                    <div key={prediction.id} className="prediction-item">
                      <div className="prediction-item-digit">{prediction.digit}</div>
                      <div className="prediction-item-info">
                        <div className="prediction-item-confidence">
                          {(prediction.confidence * 100).toFixed(1)}%
                        </div>
                        <div className="prediction-item-time">{prediction.timestamp}</div>
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
                  <li>The image is resized to 28×28 pixels.</li>
                  <li>It's converted to grayscale values.</li>
                  <li>These values are fed to a neural network (ONNX model).</li>
                  <li>The model predicts the digit!</li>
                </ul>
              </div>
            </div>

            {/* New Logs Section */}
            <div className="logs-section info-section"> {/* Re-use info-section for similar styling */}
              <button
                className="info-toggle"
                onClick={() => setShowLogs(!showLogs)}
                aria-expanded={showLogs}
              >
                Prediction Logs (Top 4)
                <span className={`info-toggle-icon ${showLogs ? 'expanded' : ''}`}>
                  ▼
                </span>
              </button>
              <div className={`info-content ${showLogs ? 'expanded' : ''}`}>
                {top4Predictions.length > 0 ? (
                  <ul className="info-list">
                    {top4Predictions.map((pred, index) => (
                      <li key={index}>
                        Digit {pred.digit}: {(pred.probability * 100).toFixed(2)}%
                      </li>
                    ))}
                  </ul>
                ) : (
                  <p>No prediction data to display. Make a prediction first.</p>
                )}
              </div>
            </div>
          </aside>
        </main>
      </div>
    </div>
  );
}

export default App;
