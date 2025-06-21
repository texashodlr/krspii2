"""
This script is the preprocessing code for krspii2.
We'll take raw PDFs in this format
1. Extraction
2. Cleaning
3. Tokenization
4. Storage
"""
import pdfplumber
import json
from transformers import AutoTokenizer
import os
from huggingface_hub import login

def token_input(token_file = 'banana.txt'):
    with open(token_file, 'r') as file:
        token = file.readline().strip()
    return token

def preprocess_pdf(pdf_path, output_path, HF_Token, tokenizer_name="mistralai/Mixtral-8x7B-v0.1"):
    tokenize = AutoTokenizer.from_pretrained(tokenizer_name, token=HF_Token)
    with pdfplumber.open(pdf_path) as pdf:
        text = ""
        for page in pdf.pages:
            text += page.extract_text() or ""

    text = text.replace("\n"," ").strip()

    tokens = tokenize(text, truncation=True, max_length=512)["input_ids"]

    with open(output_path, "w") as f:
        json.dump({"text": tokens}, f)
        f.write("\n")

HF_Token = token_input()
print(HF_Token)

# login()
input_dir  = "/data/pdfs"
os.makedirs(input_dir, exist_ok=True)
output_dir = "/data/processed"
os.makedirs(output_dir, exist_ok=True)


for pdf_file in os.listdir(input_dir):
    full_path = os.path.join(input_dir, pdf_file)
    if os.path.isfile(full_path) and pdf_file.lower().endswith(".pdf"):
        preprocess_pdf(
                os.path.join(input_dir, pdf_file),
                os.path.join(output_dir,f"{pdf_file}.jsonl"),
                HF_Token
        )
