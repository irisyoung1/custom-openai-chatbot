
# Custom OpenAI Chatbot
#### **Udacity: Build Your Own Custom Chatbot**

## Overview
This project demonstrates a custom chatbot using OpenAI's API and a dataset of 2023 fashion trends. The notebook guides you through data cleaning, generating embeddings, and interacting with the chatbot.

## How It Works
- Loads and cleans the fashion trends CSV (`data/source/2023_fashion_trends.csv`).
- Generates text embeddings using OpenAI's API.
- Lets you ask questions about fashion trends via a chatbot interface.

## Project Structure
- `project.ipynb`: Main notebook with all steps and chatbot.
- `utils.py`: Utility functions for cleaning data, generating embeddings, and answering questions.
- `requirements.txt`: Python dependencies.
- `data/source/`: Source CSV files.
- `data/results/`: Processed data and embeddings.

## Getting Started
1. Install Python 3.9+ and required packages:
   ```bash
   pip install -r requirements.txt
   ```
2. Add your OpenAI API key in the `.config.env` file
    ```
    OPENAI_API_KEY=
    OPENAI_BASE_URL=
    ```
3. Run `project.ipynb` and follow the steps to interact with the chatbot.

## Usage
- Ask questions about 2023 fashion trends in the chatbot cell at the end of the notebook.
- To exit the chatbot, press Enter without typing a question.

## Notes
- All main logic is in the notebook and `utils.py`.
- Works on Windows, Mac, and Linux.

## Acknowledgments
- Tutorials from Udacity course **Build Your Own Custom Chatbot**