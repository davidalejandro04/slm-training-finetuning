import json
from datasets import Dataset
from pathlib import Path

def build_preference_pairs(input_path, output_path):
    with open(input_path, 'r', encoding='utf-8') as f:
        records = [json.loads(line) for line in f]

    pairs = []
    for ex in records:
        prompt = ex["problem"]
        chosen = ex["generated_solution"]
        rejected = f"La respuesta es: {ex['expected_answer']}."
        pairs.append({"prompt": prompt, "chosen": chosen, "rejected": rejected})

    ds = Dataset.from_list(pairs)
    split = ds.train_test_split(test_size=0.1, seed=42)
    split.save_to_disk(output_path)
    print(f"✅ Preference dataset saved to {output_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/math_es.jsonl")
    parser.add_argument("--output", default="data/pairs")
    args = parser.parse_args()
    build_preference_pairs(args.input, args.output)
