from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

class MT:
    def __init__(self, model_name="facebook/m2m100_418M"):
        self.tok = AutoTokenizer.from_pretrained(model_name)
        dev = "mps" if torch.backends.mps.is_available() else "cpu"
        self.mod = AutoModelForSeq2SeqLM.from_pretrained(model_name, torch_dtype=torch.bfloat16).to(dev).eval()

    def translate(self, src_lang, tgt_lang, text, max_new_tokens=128):
        # M2M100 primarily needs target; src_lang may help disambiguate
        enc = self.tok(text, return_tensors="pt", truncation=True).to(self.mod.device)
        out_ids = self.mod.generate(**enc, forced_bos_token_id=self.tok.get_lang_id(tgt_lang),
                                    max_new_tokens=max_new_tokens, do_sample=False)[0]
        return self.tok.decode(out_ids, skip_special_tokens=True)
