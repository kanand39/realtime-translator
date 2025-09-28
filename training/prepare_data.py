import json, os, argparse, itertools
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

M2M = "facebook/m2m100_418M"

def batched(it, n=16):
    it = iter(it)
    while True:
        b = list(itertools.islice(it, n))
        if not b: break
        yield b

def main(inp, out_train, out_val, val_ratio=0.1, max_new_tokens=128):
    os.makedirs(os.path.dirname(out_train), exist_ok=True)
    tok = AutoTokenizer.from_pretrained(M2M)
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    model = AutoModelForSeq2SeqLM.from_pretrained(M2M, torch_dtype=torch.bfloat16).to(dev).eval()

    rows = [json.loads(l) for l in open(inp, "r", encoding="utf-8")]

    import random

    # after: rows = [...]
    random.seed(42)
    rows = [r for r in rows if r.get("src") and r.get("tgt")]  # drop empties
    random.shuffle(rows)
    N = len(rows)
    if N == 0:
        raise SystemExit(f"No valid examples in {inp}")

    # choose val size
    n_val = max(1, int(N * val_ratio)) if N > 1 else 1
    if n_val >= N:         # don’t take all rows into val
        n_val = N - 1      # leave at least 1 for train

    val = rows[:n_val]
    train = rows[n_val:]

    # safety: if either split ended empty, rebalance
    if len(train) == 0 and len(val) > 0:
        train.append(val.pop())
    if len(val) == 0 and len(train) > 1:
        val.append(train.pop())

    # ...then write train/val as you already do
    def translate_one(r):
        enc = tok(r["src"], return_tensors="pt", truncation=True).to(dev)
        out_ids = model.generate(
            **enc,
            forced_bos_token_id=tok.get_lang_id(r["tgt_lang"]),
            max_new_tokens=max_new_tokens, do_sample=False
        )[0]
        return tok.decode(out_ids, skip_special_tokens=True)

    def write(split_rows, path):
        with open(path, "w", encoding="utf-8") as f:
            for r in split_rows:
                mt = translate_one(r)
                f.write(json.dumps({
                    "src": r["src"], "mt": mt, "tgt": r["tgt"],
                    "src_lang": r["src_lang"], "tgt_lang": r["tgt_lang"]
                }, ensure_ascii=False) + "\n")

    write(train, out_train); write(val, out_val)
    print("Prepared:", out_train, out_val)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default="data/parallel.jsonl")
    ap.add_argument("--train_out", default="data/train.jsonl")
    ap.add_argument("--val_out", default="data/val.jsonl")
    args = ap.parse_args()
    main(args.inp, args.train_out, args.val_out)
