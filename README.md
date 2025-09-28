# Realtime Translator (Native iOS + Server + Training)

Phone-call-style VoIP (iOS) with live **ASR → MT → Post-Edit → TTS** and captions.
Local-first stack: **Whisper (ASR), M2M100 (MT), mT5-LoRA (post-editor), Piper (TTS)**.  
TTS falls back to **Azure** or **Google** if no local voice exists.

> Note: Carrier calls cannot be intercepted on iOS. This uses VoIP/WebRTC with **CallKit** UI (supported pattern).

---

## Prereqs

- macOS (Apple Silicon recommended), Python 3.10+, Node 18+, Xcode 15+.
- **LiveKit** (Cloud or self-host) → `LK_URL`, `LK_API_KEY`, `LK_API_SECRET`.
- (Optional) **Piper** TTS + voices: `brew install piper` and download `.onnx` voices
