# Transformer Model for Spanish to English Translation

Welcome to this project where we build a Transformer model from scratch to handle translations from Spanish to English. Dive into the world of neural machine translation with our comprehensive codebase and detailed file structure.

## Project Overview

This repository contains all the necessary components to understand, train, and infer translations using our custom-built Transformer model.

## 📂 File Structure

- **data folder**: 
  - Contains the raw text files essential for our translation tasks.

- **text vectorizer folder**: 
  - `input vectorizer`: Responsible for text vectorization of the input before passing it to the Transformer.
  - `target vectorizer`: Vectorizes target sequences ensuring compatibility with the model's outputs.

- **weights folder**: 
  - Stores the pre-trained weights of the Transformer model. These weights represent the crux of our trained machine translation system.

- **inference.py**: 
  - The go-to script for running inferences on new Spanish sentences and witnessing the Transformer's translation capabilities in action.

- **model.py**: 
  - Provides a deep dive into the structure of our Transformer model, detailing each layer and its functionality.

- **neural machine translation.ipynb**: 
  - This Jupyter notebook encompasses the entire process of data preprocessing and model training. It provides a step-by-step walkthrough, ensuring clarity at every stage of the machine translation process.

## 🚀 Getting Started

1. Ensure you have all the necessary libraries and dependencies installed.
2. Run the `neural machine translation.ipynb` notebook to train the model or leverage the pre-trained weights in the `weights folder` for quick inferences.
3. Utilize the `inference.py` script to test translations on custom sentences.

## 📝 Conclusion

Embark on this exciting journey into the realms of neural machine translation with us. Your feedback, contributions, and suggestions are always welcome!

