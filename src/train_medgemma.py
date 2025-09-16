import argparse
from io import BytesIO
from PIL import Image
from google.cloud import storage

from typing import Any, Dict
from peft import get_peft_model, LoraConfig
from peft.peft_model import PeftModel
from transformers import AutoModelForCausalLM, AutoProcessor, TrainingArguments, Trainer
from trl import SFTTrainer, SFTConfig
from datasets import load_dataset
import torch

import logging
import yaml
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

argParser = argparse.ArgumentParser()

argParser.add_argument("--config_file_path", type=str)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

def load_config(path: str) -> dict:
    """
    Load a YAML configuration file.

    Args:
        path (str): Path to the YAML file.

    Returns:
        dict: Configuration parameters.
    """
    with open(path, "r") as file:
        config = yaml.safe_load(file)
    return config

def load_image_from_gcs(gcs_uri):
    try:
        bucket_name = gcs_uri.split('gs://')[1].split('/')[0]
        blob_name = '/'.join(gcs_uri.split('gs://')[1].split('/')[1:])
        storage_client = storage.Client()
        bucket = storage_client.get_bucket(bucket_name)
        blob = bucket.blob(blob_name)
        image_bytes = blob.download_as_bytes()
        return Image.open(BytesIO(image_bytes)).convert("RGB")
    except Exception as e:
        print(f"Error loading image from GCS {gcs_uri}: {e}")
        return None


def load_model(args: Dict[str, Any]) -> PeftModel:
    model = AutoModelForCausalLM.from_pretrained(
        args["base_model_id"]
    )

    if args["LoRa"]:
        print("Initializing LoRa adapter")

        lora_config = LoraConfig(
            r=args["lora_rank"],
            lora_alpha=args["lora_rank"] * 2,
            lora_dropout=0.1,
            bias="none",
            init_lora_weights=True,
            target_modules=["q_proj", "v_proj", "o_proj", "k_proj", "gate_proj", "up_proj", "down_proj"]
        )
        model = get_peft_model(model, lora_config)

        model.print_trainable_parameters()

    return model


def format_message(text):
    messages = [
        {
            "role": "system",
            "content": [{"type": "text", "text": "You are an expert in biology."}]
        },
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Can you explain this image?"},
                {"type": "image"}
            ]
        },
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": text}
            ]
        }
    ]
    return messages


class CustomMultimodalDataCollator:
    def __init__(self, processor):
        self.processor = processor
        self.assistant_token = self.processor.tokenizer.convert_tokens_to_ids("<start_of_turn>model")
        self.prefix = "gs://fine_tuning_medgemma_week1/Gemma Hackathon/"

    def __call__(self, examples):
        texts = [self.processor.apply_chat_template(format_message(example["text"]), tokenize=False, add_generation_prompt=False) for example in examples]
        images = [example["image"].convert("RGB") for example in examples]



        
        batch = self.processor(
            text=texts,
            images=images,  
            return_tensors="pt",
            padding=True,
            truncation=True
        )

        labels = batch["input_ids"].clone()

        labels[labels == self.processor.tokenizer.pad_token_id] = -100
        batch["labels"] = labels
        return batch


def train(args):
    model = load_model(args)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    processor = AutoProcessor.from_pretrained(args["base_model_id"])
    data_collator = CustomMultimodalDataCollator(processor)


    raw_dataset = load_dataset('sachkho/gemma_data',split='train')
    # print(raw_dataset)
    # processed_dataset = raw_dataset.map(process_example,fn_kwargs={"processor": processor}).filter(lambda x: x is not None and x.get('image') is not None)
    # print(processed_dataset)
    training_args = TrainingArguments(
        output_dir=args["output_dir"],
        push_to_hub_model_id="CellGemma-3B-it",
        num_train_epochs=args["num_train_epochs"],
        per_device_train_batch_size=args["per_device_train_batch_size"],
        gradient_accumulation_steps=args["gradient_accumulation_steps"],
        learning_rate=args["learning_rate"],
        logging_steps=100,
        save_steps=100,
        save_total_limit=2,
        push_to_hub=True,
        fp16=False,
        bf16=True,
        report_to="none",
        warmup_ratio=0.1,
        optim="adamw_torch",
        disable_tqdm=False,
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        train_dataset=raw_dataset,
        args=training_args,
        data_collator=data_collator,
    )

    # --- Train the Model ---
    print("Starting training...")
    trainer.train()
    print("Training complete.")

    trainer.save_model(args["output_dir"])
    print(f"Fine-tuned model saved to {args['output_dir']}")

    processor.save_pretrained(args["output_dir"])
    print(f"Processor saved to {args['output_dir']}")

if __name__ == "__main__":

    parsed_args = argParser.parse_args()

    config_args = load_config(parsed_args.config_file_path)


    torch.manual_seed(config_args["random_seed"])
    torch.cuda.manual_seed(config_args["random_seed"])
    
    print("####################")
    for key, value in config_args.items(): 
        print(f"{key}: {value}")
    print("####################")

    train(config_args)