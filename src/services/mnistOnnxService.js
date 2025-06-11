// Global ort is loaded via UMD script in index.html
// import * as ort from 'onnxruntime-web'; // Removed ES module import

const getOrt = () => {
  // Access the global ort object loaded by UMD script
  if (typeof window !== 'undefined' && window.ort) {
    return window.ort;
  }
  throw new Error("ONNX Runtime not found. Ensure ort.min.js is loaded via script tag.");
};

const setWasmPaths = () => {
  const ort = getOrt();
  
  // Log the ort object to help diagnose its structure
  console.log("Inspecting global 'ort' object:", ort);

  if (typeof ort === 'undefined' || ort === null) {
    console.error("'ort' (ONNX Runtime) object is undefined or null. Cannot set WASM paths.");
    throw new Error("'ort' (ONNX Runtime) object is undefined or null. Ensure onnxruntime-web is loaded correctly.");
  }

  if (typeof ort.env === 'undefined' || ort.env === null) {
    console.error("'ort.env' is undefined or null. This is required for WASM configuration. The 'ort' object received was:", ort);
    throw new Error("'ort.env' is undefined. Cannot configure WASM paths. Check library initialization.");
  }

  if (typeof ort.env.wasm === 'undefined' || ort.env.wasm === null) {
    console.error("'ort.env.wasm' is undefined or null. This is critical for WASM setup. 'ort.env' received was:", ort.env);
    throw new Error("'ort.env.wasm' is undefined. Cannot configure WASM paths.");
  }

  // Vite provides `import.meta.env.BASE_URL` which will be '/' for dev and '/REPO_NAME/' for production build
  const base = import.meta.env.BASE_URL || '/';
  
  // Ensure the base path ends with a slash if it's not just "/"
  const publicPath = base.endsWith('/') ? base : `${base}/`;

  // Define the expected WASM file names.
  // IMPORTANT: If you set `ort.env.wasm.prefix` to your own `publicPath`,
  // you MUST copy these WASM files from `node_modules/onnxruntime-web/dist/`
  // into your project's `public/` folder. Vite will then copy them to your build output directory.
  // Otherwise, ONNX Runtime will fail to load them.
  const wasmFileNames = [
    'ort-wasm.wasm',
    'ort-wasm-simd.wasm',
    'ort-wasm-threaded.wasm',
    'ort-wasm-simd-threaded.wasm'
    // Check the specific version of onnxruntime-web you are using for the exact list of WASM files.
    // Some versions might also include '.jsep' files for threaded WASM.
  ];

  const wasmPaths = {};
  wasmFileNames.forEach(fileName => {
    wasmPaths[fileName] = `${publicPath}${fileName}`;
  });
  
  ort.env.wasm.wasmPaths = wasmPaths;
  // Setting `prefix` tells ONNX Runtime to look for the WASM files at this specific path.
  ort.env.wasm.prefix = publicPath;

  console.log('ONNX Runtime Web WASM Paths set to:', ort.env.wasm.wasmPaths);
  console.log('ONNX Runtime Web WASM Prefix set to:', ort.env.wasm.prefix);
  console.log(`Reminder: Ensure WASM files (e.g., ort-wasm.wasm) are present in your deployment at '${publicPath}'. If not, copy them from 'node_modules/onnxruntime-web/dist/' to your 'public/' folder before building.`);
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

export const initializeMnistModel = async (modelFilename = 'model.onnx', retries = 2, delay = 1000) => {
  setWasmPaths(); // Call this first to ensure paths are set before any ORT operation.
  try {
    console.log(`Initializing MNIST ONNX model (${modelFilename})...`);
    const modelUri = await getMnistModelUri(modelFilename);
    if (!modelUri) {
      throw new Error(`Failed to get model URI for ${modelFilename}. Ensure ${modelFilename} is in public/models/ folder.`);
    }

    const options = {
      executionProviders: ['wasm'], // 'wasm' is a good default for web. 'webgl' is another option.
      graphOptimizationLevel: 'all' // Or 'extended', 'basic'
    };

    const ort = getOrt();
    const session = await ort.InferenceSession.create(modelUri, options);
    console.log(`MNIST ONNX model (${modelFilename}) loaded successfully.`);
    URL.revokeObjectURL(modelUri); // Clean up the blob URL after loading

    console.log(`MNIST ONNX Session (${modelFilename}) Input Names:`, session.inputNames);
    console.log(`MNIST ONNX Session (${modelFilename}) Output Names:`, session.outputNames);
    
    return session; // Return the session

  } catch (error) {
    console.error(`Failed to initialize MNIST ONNX model (${modelFilename}) on this attempt:`, error);
    if (retries > 0) {
      console.log(`Retrying model initialization for ${modelFilename} in ${delay / 1000}s... (${retries} retries left)`);
      await new Promise(resolve => setTimeout(resolve, delay));
      // Recursively call with one less retry and potentially increased delay (optional: delay * 2 for exponential backoff)
      return initializeMnistModel(modelFilename, retries - 1, delay); 
    } else {
      console.error(`All retries failed for initializing model ${modelFilename}.`);
      throw error; // Propagate the error after all retries are exhausted
    }
  }
};
// Initialize autoencoder ONNX session
export const initializeAutoencoderModel = async (modelFilename = 'autoencoder.onnx') => {
  // reuse same initialize logic
  return initializeMnistModel(modelFilename);
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
    const ort = getOrt();
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
