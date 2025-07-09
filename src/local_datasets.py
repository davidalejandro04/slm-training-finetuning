from datasets import load_dataset, Dataset
import json, random

def load_jsonl(path, split=.9, seed=42):
    rows = [json.loads(l) for l in open(path, encoding="utf-8")]
    random.Random(seed).shuffle(rows)
    ds = Dataset.from_list(rows)
    return ds.train_test_split(test_size=1 - split, seed=seed)
