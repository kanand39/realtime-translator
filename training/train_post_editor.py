# training/train_post_editor.py

from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
)
from peft import LoraConfig, get_peft_model
import transformers, torch, os, inspect, sys

MODEL = "google/mt5-small"
OUT = "peft_mt5_postedit"

def format_example(r):
    # prompt format the post-editor learns to "fix" MT output
    return (
        f"src_lang: {r['src_lang']}\n"
        f"tgt_lang: {r['tgt_lang']}\n"
        f"source: {r['src']}\n"
        f"mt: {r['mt']}\n"
        f"fix:"
    )

def main():
    # 0) basic dataset presence check
    for p in ("data/train.jsonl", "data/val.jsonl"):
        if not os.path.exists(p) or os.path.getsize(p) == 0:
            print(f"[post-editor] '{p}' missing or empty; nothing to train. "
                  f"Run prepare_data.py or use pass-through server.", file=sys.stderr)
            sys.exit(0)

    # 1) load jsonl → DatasetDict
    ds = load_dataset("json", data_files={"train": "data/train.jsonl", "val": "data/val.jsonl"})

    # 2) tokenizer (slow version avoids protobuf requirement)
    tok = AutoTokenizer.from_pretrained(MODEL, use_fast=False)
    tok.padding_side = "right"

    # 3) device & dtype (keep fp32 for Apple Silicon/CPU stability)
    device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32

    # 4) base model + LoRA PEFT
    base = AutoModelForSeq2SeqLM.from_pretrained(MODEL)
    base = base.to(device=device, dtype=dtype)

    # typical LoRA targets for T5/MT5: q, k, v, o proj
    lora = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, target_modules=["q", "k", "v", "o"])
    model = get_peft_model(base, lora)

    # 5) tokenize mapper (return lists; collator will pad)
    def mapper(r):
        x = format_example(r)
        i = tok(x, truncation=True, max_length=256)
        y = tok(r["tgt"], truncation=True, max_length=256)
        i["labels"] = y["input_ids"]  # variable length OK; collator will pad and mask
        return i

    ds = ds.map(mapper, remove_columns=ds["train"].column_names)

    # 6) collator that pads inputs AND labels
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tok,
        model=model,
        padding="longest",
        label_pad_token_id=-100,   # standard ignored index for loss
        return_tensors="pt",
    )

    # 7) training arguments (v4/v5 compatible)
    common = dict(
        output_dir=OUT,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=2,
        learning_rate=5e-5,
        num_train_epochs=2,
        logging_steps=10,
        save_steps=200,
        save_total_limit=2,
        lr_scheduler_type="linear",
        warmup_ratio=0.03,
        report_to=[],               # disable wandb, etc.
        load_best_model_at_end=False,
    )
    try:
        # Transformers v5
        args = TrainingArguments(eval_strategy="steps", save_strategy="steps", eval_steps=200, **common)
    except TypeError:
        # Transformers v4
        args = TrainingArguments(evaluation_strategy="steps", save_strategy="steps", eval_steps=200, **common)

    print("Transformers:", transformers.__version__)
    print("TrainingArguments:", inspect.getfile(TrainingArguments))

    # 8) train
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=ds["train"],
        eval_dataset=ds["val"],
        data_collator=data_collator,
        tokenizer=tok,  # helps save proper config
    )
    trainer.train()

    # 9) save LoRA-adapted model + tokenizer
    os.makedirs(OUT, exist_ok=True)
    trainer.save_model(OUT)
    tok.save_pretrained(OUT)
    print("Saved:", OUT)

if __name__ == "__main__":
    # silence bitsandbytes GPU warning on Mac (optional)
    # try:
    #     import bitsandbytes  # noqa: F401
    # except Exception:
    #     pass
    main()
