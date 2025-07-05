import argparse, os, json
from datasets import load_from_disk
from datasets import DatasetDict
from datasets import Features, Value
from transformers import AutoTokenizer
from src.trainer_factory import build_trainer
from datasets import load_dataset
from src.evaluate import run_eval
from datasets import disable_caching; disable_caching()
from datasets import load_dataset
from trl import DPOConfig, DPOTrainer
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import torch.nn.functional as F
import numpy as np
import re
from tqdm.auto import tqdm


MODELS = [
 "HuggingFaceTB/SmolLM2-135M",
 "HuggingFaceTB/SmolLM2-135M-Instruct",
 "Qwen/Qwen3-0.6B",
 "Qwen/Qwen2.5-0.5B-Instruct",
 "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
 "google/gemma-3-1b-it",
 "meta-llama/Llama-3.2-1B"
]
def convert_to_prompt_format(example):
    prompt = example["problem"]
    chosen = example["generated_solution"]
    rejected = f"La respuesta es {example['expected_answer']}"  # forma corta
    return {"prompt": prompt, "chosen": chosen, "rejected": rejected}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="math_es.jsonl")
    ap.add_argument("--out",  default="results")
    ap.add_argument("--method", choices=["sft","ppl-gap","dpo"], default="ppl-gap")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--bsz",    type=int, default=4)
    ap.add_argument("--fp16",   action="store_true")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    # 1) load dataset
    from datasets import load_dataset
    ds = load_dataset("json", data_files=args.data)["train"]
    ds = ds.map(convert_to_prompt_format)

    ds = ds.train_test_split(test_size=0.1, seed=42)

    for model_name in MODELS:
        tag = model_name.split("/")[-1].replace(".", "_")
        # 2) tokenizer preproc
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        def tok_map(ex):
            ex["input_ids_chosen"]   = tokenizer(
                f"{ex['prompt']}\n{ex['chosen']}", truncation=True
            ).input_ids
            ex["input_ids_rejected"] = tokenizer(
                f"{ex['prompt']}\n{ex['rejected']}", truncation=True
            ).input_ids
            return ex
        ds_tok = ds.map(tok_map, remove_columns=ds["train"].column_names)

        # 3) evaluate baseline
        baseline_json = f"{args.out}/{tag}_baseline.json"
        if not os.path.exists(baseline_json):
            model = AutoModelForCausalLM.from_pretrained(
                model_name, torch_dtype="auto", device_map="auto",
                trust_remote_code=True)
            run_eval(model, tokenizer, ds_tok["test"], baseline_json)

        # 4) train
        trainer = build_trainer(model_name, args.method, ds_tok,
                                f"{args.out}/{tag}_{args.method}",
                                tokenizer, epochs=args.epochs,
                                bsz=args.bsz, fp16=args.fp16)
        trainer.train()
        trainer.save_model(f"{args.out}/{tag}_{args.method}")

        # 5) evaluate fine-tuned
        ft_json = f"{args.out}/{tag}_{args.method}_eval.json"
        run_eval(trainer.model, tokenizer, ds_tok["test"], ft_json)

if __name__ == "__main__":
    main()
