import * as ort from 'onnxruntime-web';

const setWasmPaths = () => {
  // Vite provides `import.meta.env.BASE_URL` which will be '/' for dev and '/REPO_NAME/' for production build
  const base = import.meta.env.BASE_URL || '/';
  
  // Ensure the base path ends with a slash if it's not just "/"
  const publicPath = base.endsWith('/') ? base : `${base}/`;

  const wasmPaths = {
    'ort-wasm.wasm': `${publicPath}ort-wasm.wasm`,
    'ort-wasm-simd.wasm': `${publicPath}ort-wasm-simd.wasm`,
    'ort-wasm-threaded.wasm': `${publicPath}ort-wasm-threaded.wasm`,
    'ort-wasm-simd-threaded.wasm': `${publicPath}ort-wasm-simd-threaded.wasm`
  };
  ort.env.wasm.wasmPaths = wasmPaths;
  // Set the prefix as well, as ORT Web might use it to locate other WASM related files or workers
  ort.env.wasm.prefix = publicPath;

  console.log('ONNX Runtime Web WASM Paths set to:', wasmPaths);
  console.log('ONNX Runtime Web WASM Prefix set to:', ort.env.wasm.prefix);
};

// Reusable function to fetch the model
const loadOnnxModelFromUrl = async (url) => {
  try {
    console.log(`Attempting to fetch ONNX model from: ${url}`);
    const response = await fetch(url, {
      method: 'GET',
      headers: {
        'Accept': 'application/octet-stream',
        'Content-Type': 'application/octet-stream' // Note: Content-Type on GET is unusual but kept as is.
      },
      cache: 'default', // Changed from 'no-cache' to 'default' for production
      mode: 'cors',
      credentials: 'same-origin'
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
  // Vite provides `import.meta.env.BASE_URL` which will be '/' for dev 
  // and '/REPO_NAME/' for production build (if `base` is set in vite.config.js).
  const base = import.meta.env.BASE_URL || '/'; 
  
  // Ensure the base path ends with a slash
  const modelPathPrefix = base.endsWith('/') ? base : `${base}/`;

  // Construct the model URL relative to the public path
  // e.g., if base is '/MNIST-Client-Side/', modelUrl will be '/MNIST-Client-Side/models/model.onnx'
  // e.g., if base is '/', modelUrl will be '/models/model.onnx'
  const localModelUrl = `${modelPathPrefix}models/${modelFilename}`;
  const fallbackModelUrl = 'https://github.com/renanbazinin/MNIST-Client-Side/raw/refs/heads/main/public/models/model.onnx';

  console.log(`Attempting to load ONNX model '${modelFilename}' from constructed local URL: ${localModelUrl}`);
  let modelBuffer = await loadOnnxModelFromUrl(localModelUrl); 

  if (!modelBuffer) {
    console.warn(`Failed to load ONNX model from local URL: ${localModelUrl}. Attempting fallback from: ${fallbackModelUrl}`);
    modelBuffer = await loadOnnxModelFromUrl(fallbackModelUrl);
  }

  if (modelBuffer) {
    try {
      const blob = new Blob([modelBuffer], { type: 'application/octet-stream' });
      const modelBlobUrl = URL.createObjectURL(blob);
      console.log(`Created ONNX model blob URL from successfully fetched model.`);
      return modelBlobUrl;
    } catch (error) {
      console.error(`Error creating blob URL for model:`, error);
      // Fall through to return null if blob creation fails
    }
  }

  // This message will be shown if both local and fallback attempts failed
  console.error(`Failed to load ONNX model '${modelFilename}'. Tried local URL: ${localModelUrl} and fallback URL: ${fallbackModelUrl}. Please ensure the model is accessible via one of these paths and the 'base' path in 'vite.config.js' is correctly set for your deployment environment if using local path.`);
  return null; 
};

export const initializeMnistModel = async () => {
  setWasmPaths(); // Call this first to ensure paths are set before any ORT operation.
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
