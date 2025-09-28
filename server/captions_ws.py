# server/captions_ws.py
import asyncio, json
import websockets

clients = set()

async def handler(ws):
    clients.add(ws)
    try:
        async for _ in ws:
            pass
    finally:
        clients.discard(ws)

async def send_caption(identity: str, text: str):
    msg = json.dumps({"to": identity, "text": text})
    dead = []
    for ws in clients:
        try:
            await ws.send(msg)
        except Exception:
            dead.append(ws)
    for ws in dead:
        clients.discard(ws)

async def run_ws():
    # Await the server and keep the task alive forever
    async with websockets.serve(handler, "0.0.0.0", 8765):
        print("captions ws listening on :8765")
        await asyncio.Future()  # run forever

if __name__ == "__main__":
    asyncio.run(run_ws())
