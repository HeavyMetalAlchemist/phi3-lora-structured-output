# scripts/train.py

import os
import json
import re
import argparse
import boto3
import wandb
import jsonlines
from pathlib import Path
from dotenv import load_dotenv
from datasets import Dataset
from peft import LoraConfig, TaskType, prepare_model_for_kbit_training, get_peft_model


load_dotenv()

# --- Args ---
parser = argparse.ArgumentParser()
parser.add_argument("--s3-bucket",     type=str, default=os.getenv("S3_BUCKET"))
parser.add_argument("--s3-prefix",     type=str, default="lora-project")
parser.add_argument("--wandb-project", type=str, default="lora-finetune")
parser.add_argument("--rank",          type=int,   default=16)
parser.add_argument("--alpha",         type=int,   default=32)
parser.add_argument("--dropout",       type=float, default=0.05)
parser.add_argument("--epochs",        type=int,   default=3)
parser.add_argument("--batch-size",    type=int,   default=4)
parser.add_argument("--grad-accum",    type=int,   default=4)
parser.add_argument("--lr",            type=float, default=2e-4)
parser.add_argument("--max-seq-len",   type=int,   default=256)
args = parser.parse_args()

S3_BUCKET  = args.s3_bucket
S3_PREFIX  = args.s3_prefix
MODEL_NAME = "microsoft/Phi-3-mini-4k-instruct"

SYSTEM_PROMPT = """You are a strict data labeling assistant.
Given a text, return ONLY valid JSON with exactly these fields:
{
  "emotion"   : "sadness | joy | love | anger | fear | surprise",
  "intensity" : "low | medium | high",
  "formality" : "formal | informal",
  "actionable": true | false
}
No markdown. No explanation. No extra fields."""


# --- S3 ---
s3_client = boto3.client("s3")

def download_from_s3(s3_key, local_path):
    s3_client.download_file(S3_BUCKET, s3_key, str(local_path))
    print(f"Downloaded s3://{S3_BUCKET}/{s3_key} → {local_path}")

def upload_to_s3(local_path, s3_key):
    s3_client.upload_file(str(local_path), S3_BUCKET, s3_key)
    print(f"Uploaded {local_path} → s3://{S3_BUCKET}/{s3_key}")


# --- Model ---
def load_model():
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

    bnb_cfg = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_cfg,
        device_map="auto",
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
    )
    return tokenizer, model


# --- Format dataset for SFTTrainer ---
def format_dataset(examples, tokenizer):
    formatted = []
    for ex in examples:
        label = json.dumps({
            "emotion"   : ex["emotion"],
            "intensity" : ex["intensity"],
            "formality" : ex["formality"],
            "actionable": ex["actionable"],
        })
        messages = [
            {"role": "system",    "content": SYSTEM_PROMPT},
            {"role": "user",      "content": f"Text: {ex['text']}"},
            {"role": "assistant", "content": label},
        ]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )
        formatted.append({"text": text})
    return formatted


# --- Main ---
if __name__ == "__main__":
    import torch
    from peft import LoraConfig, TaskType

    Path("data").mkdir(exist_ok=True)
    Path("results").mkdir(exist_ok=True)
    Path("adapter").mkdir(exist_ok=True)

    # Download data
    for split in ["train", "val"]:
        download_from_s3(
            f"{S3_PREFIX}/data/{split}.jsonl",
            f"data/{split}.jsonl"
        )

    with jsonlines.open("data/train.jsonl") as reader:
        train_examples = list(reader)
    with jsonlines.open("data/val.jsonl") as reader:
        val_examples = list(reader)

    print(f"Train: {len(train_examples)} | Val: {len(val_examples)}")

    # Load model
    print("Loading model...")
    tokenizer, model = load_model()
    print("Model loaded.")

    tokenizer.padding_side = 'right'

    model = prepare_model_for_kbit_training(model)
    print("Model prepared for kbit training.")
    model.enable_input_require_grads()

    # LoRA config
    lora_config = LoraConfig(
        r=args.rank,
        lora_alpha=args.alpha,
        lora_dropout=args.dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=["qkv_proj", "o_proj", "gate_up_proj", "down_proj"],
    )

    # Apply LoRA
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # W&B
    wandb.login(key=os.getenv("WANDB_API_KEY"))
    wandb.init(
        project=args.wandb_project,
        name="lora-training",
        config={
            "model"      : MODEL_NAME,
            "rank"       : args.rank,
            "alpha"      : args.alpha,
            "dropout"    : args.dropout,
            "epochs"     : args.epochs,
            "batch_size" : args.batch_size,
            "grad_accum" : args.grad_accum,
            "lr"         : args.lr,
            "max_seq_len": args.max_seq_len,
            "train_size" : len(train_examples),
            "val_size"   : len(val_examples),
        }
    )

    # Format datasets
    train_formatted = format_dataset(train_examples, tokenizer)
    val_formatted   = format_dataset(val_examples, tokenizer)

    train_dataset = Dataset.from_list(train_formatted)
    val_dataset   = Dataset.from_list(val_formatted)

    # Tokenize with labels
    from transformers import Trainer, TrainingArguments, DataCollatorForSeq2Seq

    def tokenize_function(example):
        result = tokenizer(
            example["text"],
            truncation=True,
            max_length=args.max_seq_len,
            padding=False,
        )
        result["labels"] = result["input_ids"].copy()
        return result

    train_tokenized = train_dataset.map(tokenize_function, remove_columns=["text"])
    val_tokenized   = val_dataset.map(tokenize_function, remove_columns=["text"])

    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True,
        pad_to_multiple_of=8,
    )

    # Training config
    training_args = TrainingArguments(
        output_dir="adapter",
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=2e-4,
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        report_to="wandb",
        bf16=False,
        fp16=True,
        dataloader_num_workers=0,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_tokenized,
        eval_dataset=val_tokenized,
        data_collator=data_collator,
    )

    print("\nStarting LoRA training...")
    trainer.train()
    print("Training complete.")

    # Save adapter
    trainer.save_model("adapter")
    tokenizer.save_pretrained("adapter")
    print("Adapter saved.")

    # Upload adapter to S3
    for f in Path("adapter").iterdir():
        if f.is_file():
            upload_to_s3(str(f), f"{S3_PREFIX}/adapter/{f.name}")

    wandb.finish()
    print("\nDone. Training complete.")