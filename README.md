# Transformer Model for Spanish to English Translation

Welcome to this project where I build a Transformer model from scratch to handle translations from Spanish to English.
## Project Overview

This repository contains all the necessary components to understand, train, and infer translations using my custom-built Transformer model.

## 📂 File Structure

- **data folder**: 
  - Contains the raw text files essential for my translation tasks.

- **text vectorizer folder**: 
  - `Input vectorizer`: Responsible for text vectorization of the input before passing it to the Transformer.
  - `Target vectorizer`: Responsible for text vectorization of the output before passing it to the Transformer.

- **weights folder**: 
  - Stores the pre-trained weights of the Transformer model. These weights represent the crux of my trained machine translation system.

- **inference.py**: 
  - The go-to script for running inferences on new Spanish sentences and witnessing the Transformer's translation capabilities in action.

- **model.py**: 
  - Provides a deep dive into the structure of my Transformer model, detailing each layer and its functionality.

- **neural machine translation.ipynb**: 
  - This Jupyter notebook encompasses the entire process of data preprocessing and model training. It provides a step-by-step walkthrough, ensuring clarity at every stage of the machine translation process.

