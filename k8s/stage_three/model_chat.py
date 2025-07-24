import os
import argparse
import logging
import subprocess
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from peft import PeftModel
import torch
import gradio as gr

# Importing model and adapters from previous pod which fine-tuned the model
base_model = "mistralai/Mistral-7B-v0.1"
adapter_path = "/data/pdfs/model"
use_bfloat16 = True


# Logging -- as per the devops book!
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Loading the tokenizer
tokenizer = AutoTokenizer.from_pretrained(adapter_path)

# Load quantized base model
quant_config = BitsAndBytesConfig(
    load_in_4bits=True,
    bnb_4bit_compute_dtype=torch.bfloat16 if use_bfloat16 else torch.float16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
)

base = AutoModelForCausalLM.from_pretrained(
    base_model,
    device_map="auto",
    quantization_config=quant_config,
    torch_dtype=torch.bfloat16 if use_bfloat16 else torch.float16,
)

# Loading the LoRA Adapater
model = PeftModel.from_pretrained(base, adapter_path)
model.eval()


def chat(user_input, history):
    conversation = ""
    for step in history:
        conversation += f"User: {step[0]}\nBrother: {step[1]}\n"
    conversation += f"User: {user_input}\nBrother:"

    inputs = tokenizer(conversation, return_tensors="pt").to("cuda")

    # Generating the output
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=150,
            do_sample=True,
            temperature=0.7,
            top_p=0.95,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
        )
    # Model's response only
    # Decode and print
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)

    # Extracting his answer
    reply = generated_text[len(conversation):].strip()
    return reply


# Launching the gradio application, defaults: localhost:7860
gr.ChatInterface(fn=chat, title="The Big Dog Discourse").launch(
    server_name="0.0.0.0", server_port=7860, share=True
)
