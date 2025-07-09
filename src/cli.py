import argparse, os, json
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from trainer_factory import build_trainer
from collators import PairCollator
from evaluate import run_eval

MODELS = [
 "HuggingFaceTB/SmolLM2-135M",
 "HuggingFaceTB/SmolLM2-135M-Instruct",
 "Qwen/Qwen3-0.6B",
 "Qwen/Qwen2.5-0.5B-Instruct",
 "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
 "google/gemma-3-1b-it",
 "meta-llama/Llama-3.2-1B"
]

# ------------------------ util ---------------------------------
def ensure_prompt_fields(ds):
    if {"prompt", "chosen", "rejected"} <= ds["train"].column_names:
        return ds  # ya OK
    # → dataset crudo tipo GSM8K / math
    def _convert(ex):
        prompt = ex.get("problem") or ex.get("context") or ""
        chosen = ex.get("generated_solution") or ex.get("chosen") or ""
        rejected = ex.get("generated_secondary_answer") or ex.get("rejected") \
                   or f"La respuesta es {ex.get('expected_answer','')}"
        return {"prompt": prompt, "chosen": chosen, "rejected": rejected}
    return ds.map(_convert)

