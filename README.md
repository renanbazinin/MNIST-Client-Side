# MNIST Digit Recognition - Client Side

🔗 **[Live Demo](https://renanbazinin.github.io/MNIST-Client-Side/)**

A simple web application to recognize handwritten digits (0-9) drawn on a canvas using a client-side ONNX model. Draw a digit and let the AI predict it!

*This project is part of an Introduction to Machine Learning course at Tel Aviv Yafo Academic College.*

This project was built with React and Vite.

## Features

*   Draw digits (0-9) on a 280x280 canvas.
*   Client-side image preprocessing:
    *   Bounding box detection and cropping to center the digit.
    *   Scaling the digit to 20x20 within a 28x28 frame.
    *   Conversion to grayscale.
    *   Normalization of pixel values to [0.0, 1.0].
*   Real-time digit prediction using an ONNX model running in the browser (via ONNX Runtime Web).
*   Displays the predicted digit and confidence level.
*   Shows the top 4 predictions with their softmax probabilities.
*   Collapsible sections for "How it works", "Recent Predictions", and "Prediction Logs".
*   Responsive design.

## Project Setup

### Prerequisites

*   Node.js and npm (or yarn)

### Installation

1.  Clone the repository:
    ```bash
    git clone https://github.com/renanbazinin/MNIST-Client-Side.git
    cd MNIST-Client-Side
    ```
2.  Install dependencies:
    ```bash
    npm install
    ```
    (or `yarn install`)

## Available Scripts

### Running the Development Server

To start the Vite development server:

```bash
npm run dev
```

This will typically open the application at `http://localhost:5173`.

### Building for Production

To create a production build in the `dist` folder:

```bash
npm run build
```

### Deploying to GitHub Pages

The project is configured for easy deployment to GitHub Pages.

1.  Ensure your `vite.config.js` has the correct `base` path (e.g., `/MNIST-Client-Side/`).
2.  Run the deploy script:
    ```bash
    npm run deploy
    ```
    This script will first build the project and then push the `dist` directory to the `gh-pages` branch of your repository.

## Model

The ONNX model (`model.onnx`) should be placed in the `public/models/` directory. This application expects it to be there to load it for inference.

## How It Works

1.  The user draws a digit on the HTML5 canvas.
2.  The drawn image is preprocessed:
    *   The canvas content is analyzed to find the bounding box of the drawn digit.
    *   The digit is cropped from the original canvas.
    *   The cropped digit is scaled to fit within a 20x20 pixel area and centered on a new 28x28 pixel canvas with a black background. Image smoothing is disabled to maintain sharpness.
    *   The 28x28 image is converted to grayscale.
    *   Pixel values are normalized to the [0.0, 1.0] range.
    *   The 28x28 image is flattened into a 784-element vector.
3.  This vector is fed as input to the pre-trained MNIST ONNX model loaded using ONNX Runtime Web.
4.  The model outputs probabilities for each digit (0-9).
5.  The application displays the digit with the highest probability as the prediction.
