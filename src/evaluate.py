import numpy as np, torch, re, json
from tqdm.auto import tqdm

def extract_num(txt):
    m = re.search(r"[-+]?\d*\.?\d+", txt); return m.group(0) if m else None

@torch.inference_mode()
def perplexity(model, tok, text):
    enc = tok(text, return_tensors="pt").to(model.device)
    loss = model(**enc, labels=enc.input_ids).loss
    return torch.exp(loss).item()

def run_eval(model, tok, ds, outfile):
    res = []
    for ex in tqdm(ds, desc="Eval"):
        p = ex["prompt"]; ch = ex["chosen"]; rj = ex["rejected"]
        ppl_g = perplexity(model, tok, f"{p}\n{ch}")
        ppl_a = perplexity(model, tok, f"{p}\n{rj}")
        test_prompt = f"Eres un tutor. Da solo la respuesta final:\n{p}"
        gen = model.generate(**tok(test_prompt, return_tensors="pt").to(model.device),
                             max_new_tokens=32)
        pred = extract_num(tok.decode(gen[0], skip_special_tokens=True))
        gold = extract_num(rj)
        res.append({"ppl_g":ppl_g,"ppl_a":ppl_a,"gap":ppl_a-ppl_g,
                    "em":pred==gold})
    # resumen
    summary = {
        "ppl_guidance": np.mean([r["ppl_g"] for r in res]),
        "ppl_answer"  : np.mean([r["ppl_a"] for r in res]),
        "ppl_gap"     : np.mean([r["gap"]   for r in res]),
        "exact_match" : np.mean([r["em"]    for r in res]),
    }
    with open(outfile, "w", encoding="utf-8") as f:
        json.dump({"records":res,"summary":summary}, f, indent=2, ensure_ascii=False)
    return summary
