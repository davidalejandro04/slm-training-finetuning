#!/usr/bin/env python3
"""
Genera un DatasetDict (train / test) con pares (prompt, chosen, rejected),
independientemente del formato original.

Detecta automáticamente:

1. **Formato “dp”**  (datasets de preferencias):
   {context, chosen, rejected, ...}

2. **Formato “gsm8k / math”** (problema + soluciones):
   {problem, generated_solution, generated_secondary_answer, ...}
   • prompt   = problem
   • chosen   = generated_solution           (explicativo)
   • rejected = generated_secondary_answer   (respuesta corta)
   (Si no hay generated_secondary_answer, usa expected_answer)

El resultado se guarda en `output_dir` con `datasets.save_to_disk()`.
Si la carpeta ya existe, se sobreescribe de forma segura.
"""

import json, random, shutil, os, argparse
from pathlib import Path
from datasets import Dataset, disable_caching

disable_caching()  # evita archivos .arrow temporales grandes


# ──────────────────────────────────────────────────────────────
def detect_format(sample: dict) -> str:
    """Devuelve 'dp' | 'gsm8k' | 'unknown'."""
    if {"context", "chosen", "rejected"} <= sample.keys():
        return "dp"
    if {"problem", "generated_solution"} <= sample.keys():
        return "gsm8k"
    return "unknown"


def convert_record(sample: dict, fmt: str):
    """Convierte al esquema {'prompt','chosen','rejected'}."""
    if fmt == "dp":
        return {
            "prompt":   sample["context"],
            "chosen":   sample["chosen"],
            "rejected": sample["rejected"],
        }
    if fmt == "gsm8k":
        rejected = sample.get("generated_secondary_answer") \
                   or f"La respuesta es {sample.get('expected_answer','')}"
        return {
            "prompt":   sample["problem"],
            "chosen":   sample["generated_solution"],
            "rejected": rejected,
        }
    raise ValueError(f"Formato no soportado: {fmt}")


def build_preference_pairs(in_file: str, out_dir: str,
                           test_split=0.1, seed=42):
    in_path  = Path(in_file)
    out_path = Path(out_dir)

    # lee primer ejemplo para detectar
    first_line = json.loads(in_path.open(encoding="utf-8").readline())
    fmt = detect_format(first_line)
    if fmt == "unknown":
        raise ValueError("No pude detectar el formato del dataset.")

    # carga y convierte todos
    records = []
    with in_path.open(encoding="utf-8") as f:
        for line in f:
            ex = json.loads(line)
            records.append(convert_record(ex, fmt))

    random.Random(seed).shuffle(records)
    split_idx = int(len(records) * (1 - test_split))
    ds = Dataset.from_list(records)
    ds = ds.train_test_split(test_size=test_split, seed=seed)

    # manejar carpeta de salida (sobrescribe sin error)
    if out_path.exists():
        shutil.rmtree(out_path)
    out_path.mkdir(parents=True, exist_ok=True)

    ds.save_to_disk(str(out_path))
    print(f"✅ Dataset ({fmt}) guardado en: {out_path} | "
          f"train={len(ds['train'])} test={len(ds['test'])}")


# ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",  required=True,
                        help="Ruta al .jsonl de entrada")
    parser.add_argument("--output", required=True,
                        help="Carpeta donde se guardará el dataset procesado")
    parser.add_argument("--test_split", type=float, default=0.1,
                        help="Proporción para test (0–1)")
    args = parser.parse_args()

    build_preference_pairs(args.input, args.output,
                           test_split=args.test_split)
