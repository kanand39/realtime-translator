// token-server/index.js (CommonJS)
const express = require('express');
const cors = require('cors');
const { AccessToken } = require('livekit-server-sdk');

const app = express();
app.use(cors());

const { LK_URL, LK_API_KEY, LK_API_SECRET, PORT = 8080, WS_HOST } = process.env;

app.get('/token', async (req, res) => {
  try {
    const identity = req.query.identity;
    const room = req.query.room || 'demo-room';
    if (!identity) return res.status(400).json({ error: 'identity required' });

    const at = new AccessToken(LK_API_KEY, LK_API_SECRET, { identity });
    // grant: room-join limited to this room
    at.addGrant({ roomJoin: true, room });

    // ✅ MUST await this
    const token = await at.toJwt();

    // Build WS URL for captions using your Mac's LAN IP
    const host = WS_HOST || (req.headers.host?.split(':')[0] ?? '127.0.0.1');
    const ws = `ws://${host}:8765`;

    res.json({ url: LK_URL, token, identity, ws });
  } catch (e) {
    console.error(e);
    res.status(500).json({ error: String(e) });
  }
});

app.listen(PORT, () => console.log(`token server on :${PORT}`));
