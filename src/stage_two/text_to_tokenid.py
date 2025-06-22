import json
import os
import argparse
import logging


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def rename_jsonl(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    for filename in os.listdir(input_dir):
        if filename.endswith(".jsonl"):
            input_path = os.path,join(input_dir, filename)
            output_path = os.path.join(output_dir, filename)
            with open(input_path, "r", encoding="utf-8") as infile, open(output_path,
                    "w", encoding="utf-8") as outfile:
                for line in infile:
                    data = json.loads(line.strip())
                    new_data = {"input_ids":data["text"]}
                    json.dump(new_data, outfile)
                    outfile.write("\n")
            logger.info(f"Processed {filename} to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Rename 'text' to 'input_ids' in JSONL files.")
    parser.add_argument("--input-dir", required=True, help="Directory containing input JSONL files.")
    parser.add_argument("--output-dir", required=True, help="Directory to save updated JSONL files.")
    args = parser.parse_args()
    rename_jsonl(args.input_dir, args.output_dir)
