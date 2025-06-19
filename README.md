# legal-llm-project
# Legal Q&A Assistant using Indian Constitutional Documents

A Retrieval-Augmented Generation (RAG) system that answers questions about Indian constitutional laws using scraped government PDFs.

## Features

- Downloads and stores Indian legal documents from [legislative.gov.in](https://legislative.gov.in) 
- Extracts text using PyMuPDF, with OCR fallback for scanned docs
- Builds semantic search index using FAISS
- Answers questions using a fine-tuned T5 model via HuggingFace Transformers
- Exposes interface through Gradio web app

## Usage

1. Clone this repo
2. Install dependencies: `pip install -r requirements.txt`
3. Run: `python app.py`
