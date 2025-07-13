import os
import argparse
import logging
import subprocess
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from peft import PeftModel
import torch

# Importing model and adapters from previous pod which fine-tuned the model
base_model = "mistralai/Mistral-7B-v0.1"
adapter_path = "../../data/model"
use_bfloat16 = True


# Logging -- as per the devops book!
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def token_input(token_file = 'banana.txt'):
    with open(token_file, 'r') as file:
        token = file.readline().strip()
    return token

def log_gpu_processes():
    """Added this in to combat 3070 OOM fails, calls nvidia-smi"""
    try:
        nvidia_smi_output = subprocess.check_output(["nvidia-smi"], text=True)
        logger.info(f"nvidia-smi output:\n{nvidia_smi_output}")
    except subprocess.CalledProcessError as e:
        logger.warning(f"Failed to run nvidia-smi: {e}")


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

# Moving to the GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Actually setting the CUDA Device
logger.info(f"Using device: {device}")
logger.info(f"GPU memory allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GiB")
logger.info(f"GPU memory reserved: {torch.cuda.memory_reserved() / 1024**3:.2f} GiB")

# Send it!
model.to(device)

# Simple Prompt
conversation = []

print("You're about to start chatting with Brother-Bot! (Type `exit` to stop!)")

while True:
    new_input = input("Speak to me, brother: ")
    if new_input.lower() in ["exit", "quit"]:
        break
    conversation.append(f"User: {new_input}")

    prompt = "\n".join(conversation) + "\nBrother-Bot:"
    
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    # Generating the output
    with torch.no_grad():
        output = model.generate(
                **inputs,
                max_new_tokens=100,
                do_sample=True,
                temperature=0.7,
                top_p=.95,
                eos_token_id=tokenizer.eos_token_id,
        )
    # Model's response only
    # Decode and print
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)

    # Extracting his answer
    reply = generated_text[len(prompt):].strip()
    print(f"Brother-Bot: {reply}")
    conversation.append(f"Brother-Bot: {reply}")

    print(f"\n++++ End Response ++++\n")
