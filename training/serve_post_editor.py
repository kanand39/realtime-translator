# training/serve_post_editor.py

from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from peft import PeftModel
import torch, uvicorn, os, re

APP = FastAPI()

MODEL_DIR = "peft_mt5_postedit"  # where train_post_editor.py saves
HAS_MODEL = os.path.isdir(MODEL_DIR) and any(os.scandir(MODEL_DIR))

TOK = None
MODEL = None

# Device selection
DEVICE = "mps" if torch.backends.mps.is_available() else (
    "cuda" if torch.cuda.is_available() else "cpu"
)

# Load model/tokenizer if fine-tuned artifacts exist
if HAS_MODEL:
    # use_fast=False avoids protobuf requirement on Mac
    TOK = AutoTokenizer.from_pretrained(MODEL_DIR, use_fast=False)
    TOK.padding_side = "right"

    base = AutoModelForSeq2SeqLM.from_pretrained("google/mt5-small")
    MODEL = PeftModel.from_pretrained(base, MODEL_DIR).to(DEVICE).eval()

class Req(BaseModel):
    src: str
    mt: str
    src_lang: str
    tgt_lang: str
    max_new_tokens: int = 80

# Regex helpers to strip artifacts
SENTINEL_RE = re.compile(r"<extra_id_\d+>")
ANGLE_TAG_RE = re.compile(r"<[^>]+>")  # catches <pad>, </s>, etc.

def sanitize(text: str) -> str:
    text = SENTINEL_RE.sub("", text)
    text = ANGLE_TAG_RE.sub("", text)
    text = " ".join(text.split()).strip()
    # trim stray leading punctuation
    text = text.lstrip(".:;-•—").strip()
    return text

@APP.post("/post_edit")
def post_edit(req: Req):
    # No fine-tuned model? return MT unchanged.
    if not HAS_MODEL or MODEL is None:
        return {"fixed": req.mt}

    # Instructional prompt; keeps output to plain text
    prompt = (
        f"src_lang: {req.src_lang}\n"
        f"tgt_lang: {req.tgt_lang}\n"
        f"source: {req.src}\n"
        f"mt: {req.mt}\n"
        f"fix: Return only the corrected translation as plain text."
    )

    enc = TOK(prompt, return_tensors="pt", truncation=True, max_length=256).to(DEVICE)
    with torch.no_grad():
        gen = MODEL.generate(
            **enc,
            max_new_tokens=min(256, max(8, req.max_new_tokens)),
            do_sample=False,
            num_beams=4,
            early_stopping=True,
        )

    # Decode & sanitize
    decoded = TOK.decode(gen[0], skip_special_tokens=True, clean_up_tokenization_spaces=True)
    fixed = sanitize(decoded)

    # Fallback if model produced empty/junk
    if not fixed or fixed in {"", ".", "<pad>"}:
        fixed = req.mt

    return {"fixed": fixed}

if __name__ == "__main__":
    # Run on all interfaces so your bot (on same Mac) can hit 127.0.0.1
    uvicorn.run(APP, host="0.0.0.0", port=8008)
