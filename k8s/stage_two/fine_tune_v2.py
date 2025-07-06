"""

JSONLs produced from Stage One, now to fine tune a model (mistral due to ability to *likely* fit on a single GPU.

Now we import the model into this script and do some Low Rank Adaptation with the PDFs we have and now we've got a model that 
can 'in theory' run a bit better in the chosen domain than prior.

Changing from Mixtral 8x7B to something smaller to fit on the 3070Ti

"""

import os
import argparse
import logging
import subprocess
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from peft import LoraConfig, get_peft_model
from datasets import load_dataset
import torch

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

def main(data_dir, output_dir, model_name, lora_rank, max_length, HF_Token):
    try:
        # Several starters to fix the Torch Kernel on a single 3070Ti
        ## Setting CUDA Device (either 4070, 3070 or 1070s)
        os.environ["PYTHONWARNINGS"]          = "ignore"
        os.environ["TOKENIZERS_PARALLELISM"]  = "false"
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
        # os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        os.environ["HUGGINGFACE_TOKEN"]       = HF_Token
        logger.info(f"Current PID: {os.getpid()}")
        log_gpu_processes()

        # Actually setting the CUDA Device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {device}")
        logger.info(f"GPU memory allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GiB")
        logger.info(f"GPU memory reserved: {torch.cuda.memory_reserved() / 1024**3:.2f} GiB")

        # Loading our tokenizer
        logger.info(f"Loading tokenizer: {model_name}")
        tokenizer = AutoTokenizer.from_pretrained(model_name, token=HF_Token)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Load PDfs (dataset)
        logger.info(f"Loading dataset from {data_dir}")
        dataset = load_dataset("json", data_files=os.path.join(data_dir, "*.jsonl"), split="train")
         
        # Quantization config
        logger.info("Configuring Quantization")
        quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
        )

        # Load model with 4-bit Quant (vRAM reasons, see stage_two_notes.txt)
        logger.info(f"Loading model: {model_name}")
        model = AutoModelForCausalLM.from_pretrained(
                model_name,
                quantization_config=quantization_config,
                device_map={"": 0}, 
                trust_remote_code=True,
                token=HF_Token,
                )

        # Configuring LoRA
        logger.info("Setting LoRA config")
        lora_config = LoraConfig(
                r=lora_rank,
                lora_alpha=16,
                target_modules=["q_proj", "v_proj"],
                lora_dropout=0.05,
                bias="none",
                task_type="CAUSAL_LM",
                )

        # Sending Model
        logger.info("Sending model to device")
        model = get_peft_model(model, lora_config)
        model.to(device)
        logger.info(f"Post-LoRA GPU Memory Allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
        torch.cuda.empty_cache()
        
        # Data collator for causal LM
        logger.info("Setting Data Collator")
        data_collator = DataCollatorForLanguageModeling(
                tokenizer=tokenizer,
                mlm=False,
                pad_to_multiple_of=8,
                )

        # Training Arguements
        log_gpu_processes()
        logger.info("Setting training arguments")
        training_args = TrainingArguments(
                output_dir=output_dir,
                per_device_train_batch_size=1,
                gradient_accumulation_steps=8,
                learning_rate=1e-4,
                num_train_epochs=1,
                max_steps=-1,
                logging_steps=10,
                save_steps=100,
                save_total_limit=2,
                fp16=True,
                remove_unused_columns=False,
                report_to="none",
                gradient_checkpointing=True,
                )
        
        # Init Trainer
        log_gpu_processes()
        logger.info("Setting the trainer")
        trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=dataset,
                data_collator=data_collator,
                processing_class=tokenizer,
                )
        
        # And start the training
        logger.info("Starting fine-tuning")
        log_gpu_processes()
        trainer.train()
        
        # Save the model
        logger.info(f"Saving the trainer model to {output_dir}")
        trainer.save_model(output_dir)
        logger.info(f"Saving the tokenizer pretrained model")
        tokenizer.save_pretrained(output_dir, token=HF_Token)

    except Exception as e:
        logger.error(f"Error during fine-tuning: {str(e)}")
        raise

if __name__ == "__main__":
    # Parser Args 
    parser = argparse.ArgumentParser(description="Fine-tuning Mistral-7B with LoRA on single consumer GPUs (XX70s)")
    parser.add_argument("--data-dir", required=True, help="Directory containing JSONL files")
    parser.add_argument("--output-dir", required=True, help="Directory to save model checkpoints")
    parser.add_argument("--model-name", default="mistralai/Mistral-7B-v0.1", help="Hugging Face Model name")
    parser.add_argument("--lora-rank", type=int, default=8, help="LoRA rank")
    parser.add_argument("--max-length", type=int, default=512, help="Maximum sequence length")
    args = parser.parse_args()
    
    HF_Token = token_input()
    logger.info(f"Token for HF: {HF_Token}")

    # Main Call
    main(args.data_dir, args.output_dir, args.model_name, args.lora_rank, args.max_length, HF_Token)
