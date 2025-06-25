"""

JSONLs produced from Stage One, now to fine tune a model (mistral due to ability to *likely* fit on a single GPU.

Now we import the model into this script and do some Low Rank Adaptation with the PDFs we have and now we've got a model that 
can 'in theory' run a bit better in the chosen domain than prior.

Changing from Mixtral 8x7B to something smaller to fit on the 3070Ti

"""

import os
import argparse
import logging
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, DataCollatorForLanguageModeling
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



def main(data_dir, output_dir, model_name, lora_rank, max_length, HF_Token):
    try:
        # Setting CUDA Device (either 4070, 3070 or 1070s)
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {device}")

        # Loading our tokenizer
        logger.info(f"Loading tokenizer: {model_name}")
        tokenizer = AutoTokenizer.from_pretrained(model_name, token=HF_Token)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Load PDfs (dataset)
        logger.info(f"Loading dataset from {data_dir}")
        dataset = load_dataset("json", data_files=os.path.join(data_dir, "*.jsonl"), split="train")

        # Load model with 4-bit Quant (vRAM reasons, see stage_two_notes.txt)
        logger.info(f"Loading model: {model_name}")
        model = AutoModelForCausalLM.from_pretrained(
                model_name,
                device_map="auto",
                load_in_4bit=True,
                torch_dtype=torch.bfloat16,
                trust_remote_code=True,
                token=HF_Token,
                )

        # Configuring LoRA
        lora_config = LoraConfig(
                r=lora_rank,
                lora_alpha=16,
                target_modules=["q_proj", "v_proj"],
                lora_dropout=0.05,
                bias="none",
                task_type="CAUSAL_LM",
                )

        # Sending Model
        model = get_peft_model(model, lora_config)
        model.to(device)
        
        # Data collator for causal LM
        data_collator = DataCollatorForLanguageModeling(
                tokenizer=tokenizer,
                mlm=False,
                )

        # Training Arguements
        training_args = TrainingArguments(
                output_dir=output_dir,
                per_device_train_batch_size=1,
                gradient_accumulation_steps=4,
                learning_rate=1e-4,
                num_train_epochs=1,
                max_steps=-1,
                logging_steps=10,
                save_steps=100,
                save_total_limit=2,
                fp16=True,
                remove_unused_columns=False,
                report_to="none",
                )
        
        # Init Trainer
        trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=dataset,
                data_collator=data_collator,
                tokenizer=tokenizer,
                )
        
        # And start the training
        logger.info("Starting fine-tuning")
        trainer.train()
        
        # Save the model
        logger.info(f"Saving the model to {output_dir}")
        trainer.save_model(output_dir)
        tokenizer.save_pretrained(output_dir)

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
