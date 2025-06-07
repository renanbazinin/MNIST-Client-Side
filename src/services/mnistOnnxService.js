import * as ort from 'onnxruntime-web';

// Reusable function to fetch the model
const loadOnnxModelFromUrl = async (url) => {
  try {
    console.log(`Attempting to fetch ONNX model from: ${url}`);
    const response = await fetch(url, {
      method: 'GET',
      headers: {
        'Accept': 'application/octet-stream',
        'Content-Type': 'application/octet-stream'
      },
      cache: 'no-cache', // Consider 'default' or 'force-cache' for production
      mode: 'cors',      // Ensure server allows CORS if fetching from different origin
      credentials: 'same-origin' // Or 'include' if needed
    });

    if (!response.ok) {
      throw new Error(`Failed to fetch model: ${response.status} ${response.statusText} from ${url}`);
    }

    console.log(`Successfully fetched model from: ${url}`);
    return await response.arrayBuffer();
  } catch (error) {
    console.error(`Error fetching model from ${url}:`, error);
    return null;
  }
};

// Function to get the URI for the MNIST model
const getMnistModelUri = async (modelFilename = 'model.onnx') => {
  const baseUrl = window.location.origin;
  let basePath = window.location.pathname;
  if (!basePath.endsWith('/')) {
    basePath = basePath.substring(0, basePath.lastIndexOf('/') + 1);
  }

  const urlsToTry = [
    new URL(`models/${modelFilename}`, baseUrl).href,
    `/models/${modelFilename}`,
    `./models/${modelFilename}`,
  ];

  if (typeof window !== 'undefined') {
     urlsToTry.push(new URL(`models/${modelFilename}`, window.location.href).href);
     // Try path relative to current page if it's in a subfolder like /app/
     if (basePath !== '/' && basePath.length > 1 && basePath.startsWith('/')) {
        // Ensure basePath is correctly formed for URL constructor if it's like /app/
        const subPath = basePath.endsWith('/') ? basePath : basePath + '/';
        urlsToTry.push(new URL(`${subPath}models/${modelFilename}`, baseUrl).href);
     }
  }

  const uniqueUrls = [...new Set(urlsToTry)];
  console.log(`Attempting to load ONNX model '${modelFilename}' from the following URLs:`, uniqueUrls);

  for (const url of uniqueUrls) {
    try {
      const modelBuffer = await loadOnnxModelFromUrl(url);
      if (modelBuffer) {
        const blob = new Blob([modelBuffer], { type: 'application/octet-stream' });
        const modelBlobUrl = URL.createObjectURL(blob);
        console.log(`Created ONNX model blob URL from successfully fetched: ${url}`);
        return modelBlobUrl;
      }
    } catch (error) {
      // Error is logged in loadOnnxModelFromUrl, continue to next URL
    }
  }

  console.error(`Failed to load ONNX model '${modelFilename}' from any of the attempted URLs.`);
  return null;
};

export const initializeMnistModel = async () => {
  // Set base path for ONNX runtime WASM files.
  // Using a CDN is generally robust.
  ort.env.wasm.wasmPaths = 'https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/';
  // Alternatively, if you serve these files locally from your public folder:
  // ort.env.wasm.wasmPaths = '/ort-wasm-files/'; // Assuming files are in public/ort-wasm-files/

  try {
    console.log("Initializing MNIST ONNX model...");
    const modelUri = await getMnistModelUri('model.onnx');
    if (!modelUri) {
      throw new Error("Failed to get model URI for model.onnx. Ensure model.onnx is in public/models/ folder.");
    }

    const options = {
      executionProviders: ['wasm'], // 'wasm' is a good default for web. 'webgl' is another option.
      graphOptimizationLevel: 'all' // Or 'extended', 'basic'
    };

    const session = await ort.InferenceSession.create(modelUri, options);
    console.log('MNIST ONNX model loaded successfully.');
    URL.revokeObjectURL(modelUri); // Clean up the blob URL after loading

    console.log("MNIST ONNX Session Input Names:", session.inputNames);
    console.log("MNIST ONNX Session Output Names:", session.outputNames);
    
    return session; // Return the session

  } catch (error) {
    console.error('Failed to initialize MNIST ONNX model:', error);
    // Propagate the error so the UI can be updated if needed
    throw error;
  }
};

export const runMnistPrediction = async (session, preprocessedInput) => {
  if (!session) {
    console.error('MNIST ONNX session not initialized.');
    return null;
  }
  if (!preprocessedInput || preprocessedInput.length !== 784) {
    console.error('Invalid input for MNIST prediction. Expected 784-element Float32Array.');
    return null;
  }

  try {
    // Ensure preprocessedInput is a Float32Array
    const inputTensor = new ort.Tensor('float32', preprocessedInput, [1, 784]);
    const feeds = { [session.inputNames[0]]: inputTensor };
    const results = await session.run(feeds);
    const outputTensor = results[session.outputNames[0]];
    return Array.from(outputTensor.data); // Return array of probabilities (logits)
  } catch (error) {
    console.error('Error during MNIST ONNX inference:', error);
    return null;
  }
};
