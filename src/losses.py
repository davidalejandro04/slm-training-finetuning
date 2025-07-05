import torch, torch.nn.functional as F

def gap_loss(model, inp):
    out_c = model(inp["chosen_ids"],   labels=inp["chosen_labels"])
    out_r = model(inp["rejected_ids"], labels=inp["rejected_labels"])
    return (out_r.loss.mean() - out_c.loss.mean())

def sft_loss(model, inp):             # clásico LoRA-SFT sobre chosen
    return model(inp["chosen_ids"], labels=inp["chosen_labels"]).loss.mean()

def dpo_loss(model, ref, beta, inp):  # versión mínima
    with torch.no_grad():
        ref_c = ref(inp["chosen_ids"]).logits
        ref_r = ref(inp["rejected_ids"]).logits
    out_c = model(inp["chosen_ids"]).logits
    out_r = model(inp["rejected_ids"]).logits
    # Δ log p sobre batch
    delta = (out_c.log_softmax(-1) - out_r.log_softmax(-1)).sum(-1)
    delta_ref = (ref_c.log_softmax(-1) - ref_r.log_softmax(-1)).sum(-1)
    return -torch.log(torch.sigmoid(beta*(delta - delta_ref))).mean()
