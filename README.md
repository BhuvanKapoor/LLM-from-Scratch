# Large Language Model from Scratch

This project implements a transformer-based language model from scratch using PyTorch, including scripts for data extraction, training, and text generation.

---

## Dataset

The dataset used in this project is the [OpenWebText dataset](https://huggingface.co/datasets/Skylion007/openwebtext) available on Hugging Face. This dataset consists of a collection of web text data scraped from various online sources, and it is commonly used for training large language models like GPT.

The raw data is processed and split into training and validation sets using the `data-extract.py` script. This script also creates the vocabulary file (`vocab.txt`) that the model uses for tokenization.

- **Dataset Source**: [OpenWebText on Hugging Face](https://huggingface.co/datasets/Skylion007/openwebtext)
- **Data Processing**: Extracted and split into training and validation datasets by the `data-extract.py` script.
- **Training/Validation Split**: 90% of the data is used for training and 10% for validation.

---

## Project Structure

```plaintext
project-root/
├── data-extract.py      # Script for extracting and processing data
├── training.py          # Model training script
├── chatbot.py           # Text generation script
├── environment.txt      # List of required packages for setting up the environment
├── model-01.pkl         # (Optional) Saved model checkpoint
└── README.md            # Project overview and instructions
```

## Getting Started
### Prerequisites
*Python 3.8 or higher
*PyTorch with CUDA support (for GPU acceleration)
*Basic understanding of transformers and language models

### Installation
1. Clone the repository:
```bash
git clone https://github.com/yourusername/LLM-From-Scratch.git
cd LLM-From-Scratch
```
2. Set up a virtual environment and install dependencies:
```bash
python -m venv llms
source llms/bin/activate  # On Windows, use llms\Scripts\activate.bat
pip install -r Environment.txt
```

3. Set up Jupyter kernel (optional):
```bash
python -m ipykernel install --user --name=llm --display-name "llm-gpt"
```
---

## Usage
1. Data Extraction
To prepare the dataset, run data-extract.py, which extracts text data from .xz files, splits it into training and validation sets, and creates a vocabulary file.
```bash
python data-extract.py
```
2. Model Training
Train the language model using training.py. This script will train the model, save checkpoints, and print the training/validation loss.
```bash
python training.py -bs 32
```
3. Text Generation
To generate text, use chatbot.py. This script loads a pre-trained model checkpoint and generates text based on user-provided prompts.
```bash
python chatbot.py -bs 32
```
Once the script starts, enter any text as a prompt to receive a generated text completion from the model.






