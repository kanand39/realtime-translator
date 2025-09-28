import os, tempfile, subprocess, re, io, wave
from typing import Callable, Optional
import requests
try:
    from google.cloud import texttospeech
except Exception:
    texttospeech = None

def _bcp47_parts(code: str):
    code = (code or "en").replace('_','-')
    m = re.match(r'^([a-zA-Z]{2,3})(?:[-_].*?([A-Z]{2}))?$', code)
    if not m: return (code.lower(), None)
    return (m.group(1).lower(), (m.group(2) or '').upper() or None)

def _chunk_wav_bytes(wav_bytes: bytes, frame_ms=20):
    f = io.BytesIO(wav_bytes)
    with wave.open(f, 'rb') as w:
        sr = w.getframerate()
        ch = w.getnchannels()
        assert ch == 1
        assert w.getsampwidth() == 2
        data = w.readframes(w.getnframes())
    bytes_per_frame = int(sr * (frame_ms/1000.0)) * 2
    for i in range(0, len(data), bytes_per_frame):
        yield data[i:i+bytes_per_frame], sr

class PiperTTS:
    def __init__(self, lang_code: str, sample_rate: int = 48000):
        self.sr = sample_rate
        self.lang_code = lang_code or "en-US"
        self.voice_path = self._pick_voice(self.lang_code)

    @staticmethod
    def _catalog():
        root = os.environ.get("PIPER_VOICE_DIR", "/usr/local/share/piper")
        if not os.path.isdir(root): return {}
        cat = {}
        for fn in os.listdir(root):
            if not fn.endswith(".onnx"): continue
            base = fn.split('.')[0]
            parts = re.split(r'[-_]', base)
            if len(parts) >= 2 and len(parts[0]) in (2,3) and len(parts[1])==2:
                bcp = f"{parts[0].lower()}-{parts[1].upper()}"
                full = os.path.join(root, fn)
                cat[bcp] = full
                cat[parts[0].lower()] = full
        return cat

    def _pick_voice(self, lang_code: str) -> Optional[str]:
        cat = self._catalog()
        lang, region = _bcp47_parts(lang_code)
        if region and f"{lang}-{region}" in cat:
            return cat[f"{lang}-{region}"]
        if lang in cat:
            return cat[lang]
        return None

    def available(self) -> bool:
        return bool(self.voice_path and os.path.isfile(self.voice_path))

    def synth_stream(self, text: str, on_frame: Callable[[bytes], None]):
        if not self.available():
            raise RuntimeError("No Piper voice for requested language")
        wav_path = tempfile.mktemp(suffix=".wav")
        subprocess.run(["piper","-m",self.voice_path,"-f",wav_path,"-s",str(self.sr)],
                       input=text.encode("utf-8"), check=True)
        with open(wav_path, "rb") as f: wav = f.read()
        os.remove(wav_path)
        for frame, _ in _chunk_wav_bytes(wav, frame_ms=20):
            on_frame(frame)

class AzureTTS:
    def __init__(self, lang_code: str, sample_rate: int = 48000):
        self.lang_code = lang_code or "en-US"
        self.sr = sample_rate
        self.key = os.environ.get("AZURE_SPEECH_KEY")
        self.region = os.environ.get("AZURE_SPEECH_REGION")
        self.voice = self._pick_voice(self.lang_code)

    @staticmethod
    def _voice_map():
        return {
            "en": "en-US-JennyNeural", "es": "es-ES-ElviraNeural", "fr": "fr-FR-DeniseNeural",
            "de": "de-DE-KatjaNeural", "hi": "hi-IN-SwaraNeural", "ja": "ja-JP-NanamiNeural",
            "ko": "ko-KR-SunHiNeural", "zh": "zh-CN-XiaoxiaoNeural", "pt": "pt-BR-FranciscaNeural",
            "ar": "ar-EG-SalmaNeural", "ru": "ru-RU-DariyaNeural", "it": "it-IT-ElsaNeural", "tr": "tr-TR-EmelNeural",
        }

    def _pick_voice(self, lang_code: str):
        lang, _ = _bcp47_parts(lang_code); m = self._voice_map()
        return m.get(lang, m["en"])

    def available(self) -> bool:
        return bool(self.key and self.region)

    def synth_stream(self, text: str, on_frame: Callable[[bytes], None]):
        if not self.available(): raise RuntimeError("Azure creds missing")
        url = f"https://{self.region}.tts.speech.microsoft.com/cognitiveservices/v1"
        ssml = f"""<speak version="1.0" xml:lang="{self.lang_code}">
          <voice name="{self.voice}">{text}</voice>
        </speak>""".strip()
        headers = {
            "Ocp-Apim-Subscription-Key": self.key,
            "Content-Type": "application/ssml+xml",
            "X-Microsoft-OutputFormat": "riff-16khz-16bit-mono-pcm",
        }
        r = requests.post(url, headers=headers, data=ssml.encode("utf-8"), timeout=30); r.raise_for_status()
        wav = r.content
        for frame, _ in _chunk_wav_bytes(wav, frame_ms=20):
            on_frame(frame)

class GoogleTTS:
    def __init__(self, lang_code: str, sample_rate: int = 16000):
        self.lang_code = (lang_code or "en-US"); self.sr = sample_rate
        self.enabled = bool(texttospeech and os.environ.get("GOOGLE_APPLICATION_CREDENTIALS"))

    def available(self) -> bool: return self.enabled

    def synth_stream(self, text: str, on_frame: Callable[[bytes], None]):
        if not self.available(): raise RuntimeError("Google creds missing")
        client = texttospeech.TextToSpeechClient()
        input_text = texttospeech.SynthesisInput(text=text)
        voice = texttospeech.VoiceSelectionParams(language_code=self.lang_code,
                                                 ssml_gender=texttospeech.SsmlVoiceGender.NEUTRAL)
        audio_config = texttospeech.AudioConfig(audio_encoding=texttospeech.AudioEncoding.LINEAR16,
                                                sample_rate_hertz=self.sr)
        resp = client.synthesize_speech(input=input_text, voice=voice, audio_config=audio_config)
        data = resp.audio_content
        frame = int(self.sr * 0.02) * 2
        for i in range(0, len(data), frame):
            on_frame(data[i:i+frame])

def get_tts(lang_code: str, sample_rate: int = 48000):
    p = PiperTTS(lang_code, sample_rate)
    if p.available(): return p
    a = AzureTTS(lang_code, sample_rate)
    if a.available(): return a
    g = GoogleTTS(lang_code, min(sample_rate, 16000))
    if g.available(): return g
    p_en = PiperTTS("en-US", sample_rate)
    if p_en.available(): return p_en
    raise RuntimeError("No TTS available. Add Piper voices or set Azure/Google creds.")
