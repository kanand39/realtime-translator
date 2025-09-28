import re
class Chunker:
    def __init__(self, k_tokens=8):
        self.buf=[]; self.k=k_tokens
    def feed(self, text):
        toks = re.findall(r"\w+|[^\w\s]", text, re.UNICODE)
        self.buf.extend(toks); chunks=[]
        while len(self.buf)>=self.k or (self.buf and self.buf[-1] in ".?!，。！？"):
            cut=len(self.buf)
            for i in range(len(self.buf)-1,-1,-1):
                if self.buf[i] in [".","?","!","，","。","！","？"]:
                    cut=i+1; break
            if cut<self.k: cut=self.k
            seg="".join(self.buf[:cut]); chunks.append(seg); self.buf=self.buf[cut:]
        return chunks
    def commit(self):
        if not self.buf: return None
        seg="".join(self.buf); self.buf=[]
        return seg
