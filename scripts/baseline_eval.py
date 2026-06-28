# scripts/baseline_eval.py

import os
import json
import re
import argparse
import boto3
import wandb
import jsonlines
from tqdm import tqdm
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# --- Args ---
parser = argparse.ArgumentParser()
parser.add_argument("--s3-bucket",     type=str, default=os.getenv("S3_BUCKET"))
parser.add_argument("--s3-prefix",     type=str, default="lora-project")
parser.add_argument("--wandb-project", type=str, default="lora-finetune")
args = parser.parse_args()

S3_BUCKET  = args.s3_bucket
S3_PREFIX  = args.s3_prefix
MODEL_NAME = "microsoft/Phi-3-mini-4k-instruct"

VALID_EMOTIONS    = {"sadness", "joy", "love", "anger", "fear", "surprise"}
VALID_INTENSITIES = {"low", "medium", "high"}
VALID_FORMALITIES = {"formal", "informal"}

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


# --- JSON ---
def extract_json(text: str):
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group())
    except json.JSONDecodeError:
        return None


def validate_schema(parsed: dict) -> bool:
    if parsed is None:
        return False
    if parsed.get("emotion")    not in VALID_EMOTIONS:
        return False
    if parsed.get("intensity")  not in VALID_INTENSITIES:
        return False
    if parsed.get("formality")  not in VALID_FORMALITIES:
        return False
    if not isinstance(parsed.get("actionable"), bool):
        return False
    return True


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
    model.eval()
    return tokenizer, model


# --- Inference ---
def run_inference(text: str, tokenizer, model) -> str:
    import torch

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": f"Text: {text}"},
    ]
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=100,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            use_cache=True,
        )

    gen_ids  = outputs[0, inputs.input_ids.shape[1]:]
    raw_text = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
    return raw_text


# --- Eval loop ---
def run_eval(test_examples, tokenizer, model, inference_fn=None):
    results = []

    correct_emotion    = 0
    correct_intensity  = 0
    correct_formality  = 0
    correct_actionable = 0
    schema_compliant   = 0

    for ex in tqdm(test_examples):
        text = ex["text"]
        gt   = {
            "emotion"   : ex["emotion"],
            "intensity" : ex["intensity"],
            "formality" : ex["formality"],
            "actionable": ex["actionable"],
        }

        if inference_fn:
            raw_output = inference_fn(text)
        else:
            raw_output = run_inference(text, tokenizer, model)

        parsed   = extract_json(raw_output)
        is_valid = validate_schema(parsed)

        if is_valid:
            schema_compliant   += 1
            em = parsed["emotion"]    == gt["emotion"]
            ii = parsed["intensity"]  == gt["intensity"]
            ff = parsed["formality"]  == gt["formality"]
            aa = parsed["actionable"] == gt["actionable"]
            correct_emotion    += int(em)
            correct_intensity  += int(ii)
            correct_formality  += int(ff)
            correct_actionable += int(aa)
        else:
            em = ii = ff = aa = False

        results.append({
            "text"              : text,
            "gt_emotion"        : gt["emotion"],
            "gt_intensity"      : gt["intensity"],
            "gt_formality"      : gt["formality"],
            "gt_actionable"     : gt["actionable"],
            "raw_output"        : raw_output,
            "pred_emotion"      : parsed.get("emotion")    if parsed else None,
            "pred_intensity"    : parsed.get("intensity")  if parsed else None,
            "pred_formality"    : parsed.get("formality")  if parsed else None,
            "pred_actionable"   : parsed.get("actionable") if parsed else None,
            "schema_compliant"  : is_valid,
            "correct_emotion"   : em,
            "correct_intensity" : ii,
            "correct_formality" : ff,
            "correct_actionable": aa,
        })

    n = len(test_examples)
    metrics = {
        "schema_compliance"  : schema_compliant   / n,
        "emotion_accuracy"   : correct_emotion    / n,
        "intensity_accuracy" : correct_intensity  / n,
        "formality_accuracy" : correct_formality  / n,
        "actionable_accuracy": correct_actionable / n,
        "overall_accuracy"   : (
            correct_emotion + correct_intensity +
            correct_formality + correct_actionable
        ) / (n * 4),
    }

    return results, metrics


# --- Main ---
if __name__ == "__main__":

    Path("data").mkdir(exist_ok=True)
    Path("results").mkdir(exist_ok=True)

    # Download test set
    download_from_s3(f"{S3_PREFIX}/data/test.jsonl", "data/test.jsonl")

    with jsonlines.open("data/test.jsonl") as reader:
        test_examples = list(reader)

    print(f"Test examples loaded: {len(test_examples)}")

    # Load model
    print("Loading model...")
    tokenizer, model = load_model()
    print("Model loaded.")

    # W&B
    wandb.login(key=os.getenv("WANDB_API_KEY"))
    wandb.init(
        project=args.wandb_project,
        name="baseline-eval",
        config={
            "model"        : MODEL_NAME,
            "test_size"    : len(test_examples),
            "quantization" : "4bit-nf4",
        }
    )

    # Run eval
    results, metrics = run_eval(test_examples, tokenizer, model)

    # Print metrics
    print("\n--- Baseline Metrics ---")
    for k, v in metrics.items():
        print(f"  {k:25s}: {v:.4f}")

    wandb.log(metrics)
    wandb.finish()

    # Save results
    with jsonlines.open("results/baseline_results.jsonl", mode="w") as writer:
        writer.write_all(results)

    with open("results/baseline_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    # Upload to S3
    upload_to_s3("results/baseline_results.jsonl", f"{S3_PREFIX}/results/baseline_results.jsonl")
    upload_to_s3("results/baseline_metrics.json",  f"{S3_PREFIX}/results/baseline_metrics.json")

    print("\nDone. Baseline Evaluation complete.")