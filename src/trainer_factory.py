from transformers import TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model
from losses import gap_loss, sft_loss, dpo_loss
from collators import PairCollator
import os

def build_trainer(model_name, method, ds, output, tokenizer,
                  lr=5e-6, epochs=3, bsz=4, beta=0.1, fp16=True):
    from transformers import AutoModelForCausalLM

    base = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype="auto",
            device_map="auto", trust_remote_code=True)
    lora_cfg = LoraConfig(r=16,lora_alpha=16,lora_dropout=.05,
                          target_modules=["q_proj","k_proj","v_proj","o_proj"])
    model = get_peft_model(base, lora_cfg)

    args = TrainingArguments(
        output_dir=output, per_device_train_batch_size=bsz,
        gradient_accumulation_steps=1, num_train_epochs=epochs,
        learning_rate=lr, logging_steps=25, evaluation_strategy="epoch",
        save_strategy="epoch", fp16=fp16, remove_unused_columns=False,
        report_to="wandb" if os.getenv("WANDB_API_KEY") else "none")

    # Selecciona la loss
    if method == "ppl-gap":
        def compute_loss(m, inp, **kw): return gap_loss(m, inp)
    elif method == "sft":
        def compute_loss(m, inp, **kw): return sft_loss(m, inp)
    elif method == "dpo":
        ref = AutoModelForCausalLM.from_pretrained(
                model_name, torch_dtype="auto", device_map="auto",
                trust_remote_code=True).eval()
        def compute_loss(m, inp, **kw): return dpo_loss(m, ref, beta, inp)
    else:
        raise ValueError("Método no soportado")

    return Trainer(model=model, args=args,
                   train_dataset=ds["train"], eval_dataset=ds["test"],
                   data_collator=PairCollator(tokenizer.pad_token_id),
                   compute_loss=compute_loss)
