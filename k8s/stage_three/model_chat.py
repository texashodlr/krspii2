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
adapter_path = "../../data/model"
use_bfloat16 = True


# Logging -- as per the devops book!
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

def token_input(token_file = 'banana.txt'):
    with open(token_file, 'r') as file:
        token = file.readline().strip()
    return token

def model_load(token):
    # Loading the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(adapter_path,token=HF_Token)
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
            token=HF_Token,
    )
    # Loading the LoRA Adapater
    model = PeftModel.from_pretrained(base, adapter_path)
    model.eval()
    return model, tokenizer

def chat(user_input, history, max_history=5):
    history = history[-max_history:] if len(history) > max_history else history
    conversation = ""
    for user_msg, assistant_msg in history:
        conversation += f"User: {user_msg}\nBrother: {assistant_msg}\n"
    conversation += f"User: {user_input}\nBrother:"
    
    logger.info(f"Conversation prompt:\n{conversation}")

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
    try:
        reply_start = generated_text.rfind("Brother:") + len("Brother:")
        reply = generated_text[reply_start:].strip()
        if not reply:
            reply = "No response generated."
    except: 
        reply = generated_text[len(conversation):].strip() or "No response generated."
    logger.info(f"Extracted reply: {reply}")
    return reply


# Tokens as usual
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Chat with fined-tuned a Big Dog!(Mistral-7b)")
    parser.add_argument("--token-file", default="banana.txt", help="Path to Hugging face token file")
    args = parser.parse_args()

    HF_Token = token_input()
    logger.info(f"Token for HF: {HF_Token}")
    model, tokenizer = model_load(HF_Token)

    # Launching the gradio application, defaults: localhost:7860
    gr.ChatInterface(fn=chat,
            title="The Big Dog Discourse",
            additional_inputs=[gr.Slider(minimum=1, maximum=10, value=5,label="Max History Length")],
    ).launch(server_name="0.0.0.0", server_port=7860, share=True)
