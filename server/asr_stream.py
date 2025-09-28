import numpy as np, threading, time
import webrtcvad
from faster_whisper import WhisperModel

class ASRStream:
    def __init__(self, sample_rate=48000, vad_frame_ms=20, lang=None, model_size="small"):
        self.model = WhisperModel(model_size, device="cpu")
        self.sample_rate = sample_rate
        self.vad = webrtcvad.Vad(2)
        self.buf = bytes()
        self.frame_bytes = int(sample_rate * vad_frame_ms / 1000) * 2
        self.listeners = {"partial":[], "final":[]}
        self.lang_hint = lang
        self._stop = False
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()


    def on(self, kind, cb): self.listeners[kind].append(cb)
    def feed(self, pcm16_bytes): self.buf += pcm16_bytes
    def stop(self): self._stop = True

    def _emit(self, kind, text, lang):
        for cb in self.listeners[kind]:
            cb(text, lang)

    def _loop(self):
        speech=False; seg=bytearray(); last_speech=time.time()
        while not self._stop:
            if len(self.buf) < self.frame_bytes:
                time.sleep(0.005); continue
            frame = self.buf[:self.frame_bytes]; self.buf = self.buf[self.frame_bytes:]
            is_speech = self.vad.is_speech(frame, self.sample_rate)
            if is_speech:
                seg.extend(frame); speech=True; last_speech=time.time()
                if len(seg) >= self.sample_rate//2*2:
                    txt, lang = self._transcribe(seg, True)
                    self._emit("partial", txt, lang)
            else:
                if speech and (time.time()-last_speech)>0.3:
                    txt, lang = self._transcribe(seg, False)
                    self._emit("final", txt, lang)
                    seg=bytearray(); speech=False

    def _transcribe(self, pcm, partial):
        audio = np.frombuffer(pcm, dtype=np.int16).astype(np.float32)/32768.0
        segments, info = self.model.transcribe(audio, language=self.lang_hint, vad_filter=False, beam_size=1, best_of=1)
        text = "".join(s.text for s in segments).strip()
        lang = getattr(info, "language", None) or "auto"
        return text, lang
    

