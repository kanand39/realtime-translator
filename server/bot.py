# server/bot.py — LiveKit Python 1.x, no agents, robust frame binding
import os, json, asyncio, aiohttp
from dotenv import load_dotenv
from livekit import rtc
import time

from asr_stream import ASRStream
from mt_m2m import MT
from chunker import Chunker
from captions_ws import send_caption, run_ws
from tts_backend import get_tts

load_dotenv()

LK_URL = os.environ.get("LK_URL", "")
ROOM = os.environ.get("ROOM", "demo-room")
POSTEDITOR_URL = os.environ.get("POSTEDITOR_URL", "http://127.0.0.1:8008/post_edit")
DEFAULT_FALLBACK = os.environ.get("DEFAULT_FALLBACK", "en")
TOKEN_SERVER = os.environ.get("TOKEN_SERVER", "http://127.0.0.1:8080/token")
BOT_IDENTITY = os.environ.get("BOT_IDENTITY", "translator-bot")

last_lang_by_identity: dict[str, str] = {}
def other_identity(x: str) -> str: return "callerB" if x.endswith("A") else "callerA"
def choose_target_for(from_id: str, src_lang: str | None) -> str:
    tgt = last_lang_by_identity.get(other_identity(from_id))
    if tgt: return tgt
    if src_lang and src_lang != "auto":
        return "es" if src_lang.startswith("en") and DEFAULT_FALLBACK == "es" else DEFAULT_FALLBACK
    return DEFAULT_FALLBACK

mt = MT()
_tts_cache: dict[tuple[str, int], object] = {}
def tts_for(lang: str, sr: int):
    k = (lang, sr)
    if k not in _tts_cache: _tts_cache[k] = get_tts(lang, sr)
    return _tts_cache[k]

async def post_edit(session: aiohttp.ClientSession, src: str, mt_text: str, src_lang: str, tgt_lang: str) -> str:
    async with session.post(POSTEDITOR_URL, json={"src": src, "mt": mt_text, "src_lang": src_lang, "tgt_lang": tgt_lang}) as r:
        js = await r.json(); return js.get("fixed", mt_text)

async def main():
    print("Bot starting…")
    asyncio.create_task(run_ws())

    async with aiohttp.ClientSession() as sess_fetch:
        async with sess_fetch.get(TOKEN_SERVER, params={"identity": BOT_IDENTITY, "room": ROOM}) as r:
            r.raise_for_status(); js = await r.json()
            lk_url = js.get("url") or LK_URL; token = js["token"]

    room = rtc.Room()
    await room.connect(lk_url, token)
    print("Joined:", ROOM)

    main_loop = asyncio.get_running_loop()

    sample_rate, num_channels = 48000, 1
    source_to_A = rtc.AudioSource(sample_rate, num_channels)
    source_to_B = rtc.AudioSource(sample_rate, num_channels)
    track_to_A = rtc.LocalAudioTrack.create_audio_track("translated_to_A", source_to_A)
    track_to_B = rtc.LocalAudioTrack.create_audio_track("translated_to_B", source_to_B)
    await room.local_participant.publish_track(track_to_A)
    await room.local_participant.publish_track(track_to_B)

    session = aiohttp.ClientSession()

    # --- robust subscription handler (works with varying signatures) ---
    def on_track(*args):
        track = None; participant = None
        if len(args) == 3:
            a, b, c = args
            if hasattr(a, "kind"): track, participant = a, c           # (track, publication, participant)
            elif hasattr(b, "kind"): track, participant = b, c         # (publication, track, participant)
            else: participant = c; track = getattr(a, "track", None)   # (publication, ?, participant)
        elif len(args) == 2:
            publication, participant = args
            track = getattr(publication, "track", None)                 # (publication, participant)
        else:
            print("track_subscribed: unexpected args len", len(args)); return

        if participant is None: print("track_subscribed: missing participant"); return

        if (track is None) or not hasattr(track, "kind"):
            try:
                for pub in participant.tracks:
                    if getattr(pub, "track", None) and getattr(pub.track, "kind", None) == rtc.TrackKind.KIND_AUDIO:
                        track = pub.track; break
            except Exception as e:
                print("scan participant tracks failed:", e)

        if track is None or track.kind != rtc.TrackKind.KIND_AUDIO:
            print("track_subscribed: no audio track for", getattr(participant, "identity", "?")); return

        print("Subscribed to", participant.identity, "| track type:", type(track).__name__)
        asyncio.create_task(
        handle_track(track, participant, source_to_A, source_to_B,
                     session, sample_rate, num_channels, main_loop)   # <— ADD main_loop
    )

    room.on("track_subscribed", on_track)

    try:
        while True: await asyncio.sleep(3600)
    finally:
        await session.close()
        try: await room.disconnect()
        except Exception: pass

async def handle_track(track, participant, source_to_A, source_to_B,
                       session, sample_rate, num_channels, main_loop):
    from_identity = participant.identity
    role = "A" if from_identity.endswith("A") else "B"
    out_src = source_to_B if role == "A" else source_to_A
    listener_identity = "callerB" if role == "A" else "callerA"

    asr = ASRStream(sample_rate=sample_rate, model_size="small")

    chunker = Chunker(k_tokens=2)  # handle short phrases like "hola"

    async def process_text(text: str, src_lang: str | None, final: bool = False):
        if not text.strip(): return
        tgt_lang = choose_target_for(from_identity, src_lang or "auto")
        tts = tts_for(tgt_lang, sample_rate)
        mt_out = mt.translate(src_lang or "auto", tgt_lang, text)
        fixed = await post_edit(session, text, mt_out, src_lang or "auto", tgt_lang)

        def on_pcm(pcm: bytes):
            out_src.capture_frame(pcm, sample_rate=sample_rate, num_channels=num_channels, bytes_per_sample=2)

        await asyncio.to_thread(tts.synth_stream, fixed, on_pcm)
        if final:
            await send_caption(listener_identity, fixed)
            print(f"[{from_identity}] {src_lang}→{tgt_lang} | '{text[:32]}' → '{fixed[:32]}'")

    last_partial_ts = 0.0  # monotonic seconds

    def on_partial(t, lang):
        nonlocal last_partial_ts
        last_partial_ts = time.monotonic()                    


    def on_final(t, lang):
        # (optional debug)
        # print(f"[final   {participant.identity}] lang={lang} text='{t}'")
        seg = chunker.commit() or t
        if lang and lang != "auto":
            last_lang_by_identity[from_identity] = lang
        asyncio.run_coroutine_threadsafe(
            process_text(seg, lang, final=True), main_loop
        )

    asr.on("partial", on_partial)
    asr.on("final", on_final)

    # Idle flush so one-word utterances finalize after brief silence
    async def _idle_flush():
        nonlocal last_partial_ts
        while True:
            await asyncio.sleep(0.2)
            if not last_partial_ts:
                continue
            now = time.monotonic() 
            if now - last_partial_ts > 0.5:  # ~500 ms silence
                seg = chunker.commit()
                if seg:
                    asyncio.run_coroutine_threadsafe(
                        process_text(seg, None, final=True), main_loop
                    )
                last_partial_ts = 0.0

    asyncio.create_task(_idle_flush())

    # --------- FRAME BINDING (try all known APIs) ----------
    # 1) Preferred: AudioStream async iterator
    try:
        from livekit.rtc import AudioStream  # may or may not exist in your build
        stream = AudioStream(track)

        async def pump_stream():
            async for ev in stream:
                # Some builds yield AudioFrameEvent with ev.frame -> AudioFrame
                frame = getattr(ev, "frame", ev)
                data = getattr(frame, "data", None)
                if data is not None:
                    asr.feed(data)

        asyncio.create_task(pump_stream())
        print("Audio frames bound via AudioStream iterator")
        return
    except Exception:
        pass

    # 2) Event name: frame_received
    try:
        @track.on("frame_received")  # requires .on API on track
        def _on_frame(frame: rtc.AudioFrame):
            asr.feed(frame.data)
        print("Audio frames bound via track.on('frame_received')")
        return
    except Exception:
        pass

    # 3) Event name: audio_frame
    try:
        @track.on("audio_frame")
        def _on_frame_alt(frame: rtc.AudioFrame):
            asr.feed(frame.data)
        print("Audio frames bound via track.on('audio_frame')")
        return
    except Exception as e:
        print("No usable frame API on this SDK build:", e)
        print(">>> If you see no frames, pin the SDK to a compatible version:")
        print(">>>    pip install 'livekit<1.0.0'  # then rerun the bot")
        return

if __name__ == "__main__":
    asyncio.run(main())
