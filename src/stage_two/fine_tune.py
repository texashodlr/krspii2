"""

JSONLs produced from Stage One, now to fine tune a model (mistral due to ability to *likely* fit on a single GPU.

Now we import the model into this script and do some Low Rank Adaptation with the PDFs we have and now we've got a model that 
can 'in theory' run a bit better in the chosen domain than prior.

Changing from Mixtral 8x7B to something smaller to fit on the 3070Ti

"""

import os
import argparse
import logging
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model
from datasets import load_dataset
import torch

# Logging -- as per the devops book!
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def main(data_dir, output_dir, model_name, lora_rank, max_length):
    try:
        # Setting CUDA Device (either 4070, 3070 or 1070s)

    except Exception as e:
        logger.error(f"Error during fine-tuning: {str(e)}")
        raise

if __name__ == "__main__":
    # Parser

    main()
