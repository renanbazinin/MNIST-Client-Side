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


## MNIST Prediction Examples
Here are some examples showing how the MNIST digit recognition model performs on user-drawn digits:

**Example 1: Digit 4**  
![Drawn 4](https://i.imgur.com/SVTWjOY.png)  
_Model Prediction: 4 (97.1% confidence)_

**Example 2: Digit 8**  
![Drawn 8](https://i.imgur.com/KN4SI34.png)  
_Model Prediction: 8 (high confidence)_



## Autoencoder

🔗 **[Autoencoder Demo](https://renanbazinin.github.io/MNIST-Client-Side/#/encoder)**

Our autoencoder model not only compresses and reconstructs handwritten digits but also effectively cleans noise from your drawings. This neural network learns to encode the key features of the digit and decode a clear, denoised version.

### Autoencoder in Action
**Input (what you draw):**  
![Noisy Input](https://i.imgur.com/zSvrVLz.png)  

**Output (denoised output):**  
![Denoised Output](https://i.imgur.com/7oOhWwp.png)

Feel free to switch between different autoencoder variations and compare reconstruction quality in the demo.




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

