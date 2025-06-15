"""
This script is the preprocessing code for krspii2.
We'll take raw PDFs in this format
1. Extraction
2. Cleaning
3. Tokenization
4. Storage
"""
#import fitz
import pdfplumber
import json
from transformers import AutoTokenizer
import os
from huggingface_hub import login

# Extraction
"""
doc = fitz.open("example.pdf")
for page in doc:
    print(page.get_text())
for page in doc:
    blocks = page.get_text("blocks")
    for block in blocks:
        print(block[4])
"""

def preprocess_pdf(pdf_path, output_path, tokenizer_name="mistralai/Mixtral-8x7B-v0.1"):
    tokenize = AutoTokenizer.from_pretrained(tokenizer_name)
    with pdfplumber.open(pdf_path) as pdf:
        text = ""
        for page in pdf.pages:
            text += page.extract_text() or ""

    text = text.replace("\n"," ").strip()

    tokens = tokenizer(text, truncation=True, max_length=512)["input_ids"]

    with open(output_path, "w") as f:
        json.dump({"text": tokens}, f)
        f.write("\n")

login()
input_dir  = "../data/pdfs"
output_dir = "../data/processed"


os.makedirs(output_dir, exist_ok=True)
for pdf_file in os.listdir(input_dir):
    preprocess_pdf(
            os.path.join(input_dir, pdf_file),
            os.path.join(output_dir,f"{pdf_file}.jsonl")
            )
