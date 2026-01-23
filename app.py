"""
app.py — Exotel Voicebot (bidirectional WS) + OpenAI Realtime + Multi-tenant SaaS base (SQLite)

This file uses your proven ws_server.py runtime patterns, and adds:
- Tenants table (called-number -> tenant)
- Admin API to manage tenants (X-Admin-Key)
- Exotel WS bootstrap per tenant (returns wss URL with called number)
- Exotel WS handler that bridges audio <-> OpenAI Realtime
- Start-on-media failsafe + WS OUT logs

Environment variables:
  PORT=10000 (Render)
  BASE_URL or PUBLIC_BASE_URL (your public domain, without scheme is ok)
  ADMIN_API_KEY (required for /admin/*)
  DATABASE_URL (optional; default sqlite:///./dev.db)

OpenAI:
  OPENAI_API_KEY (or OpenAI_Key or OPENAI_KEY)
  OPENAI_REALTIME_MODEL (default gpt-4o-realtime-preview)
  OPENAI_VOICE (default "verse")

Exotel (optional for future REST actions; not required for Voicebot streaming itself):
  EXOTEL_SID, EXOTEL_TOKEN, EXO_CALLER_ID, EXOTEL_FLOW_URL, EXO_SUBDOMAIN
"""

import os
import json
import base64
import asyncio
import logging
import datetime as dt
from typing import Optional, Dict, Any

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Depends, Header, HTTPException, Request
from fastapi.responses import PlainTextResponse
from fastapi.middleware.cors import CORSMiddleware

from sqlalchemy import create_engine, String, Integer, DateTime, Text, Boolean, ForeignKey
from sqlalchemy.orm import sessionmaker, DeclarativeBase, Session, Mapped, mapped_column, relationship

import aiohttp
from aiohttp import WSMsgType


# ---------------- Logging ----------------
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(level=getattr(logging, LOG_LEVEL, logging.INFO))
logger = logging.getLogger("app")


# ---------------- Settings ----------------
def _get_env_any(*names: str, default: str = "") -> str:
    for n in names:
        v = os.getenv(n)
        if v:
            return v
    return default

OPENAI_API_KEY = _get_env_any("OPENAI_API_KEY", "OpenAI_Key", "OPENAI_KEY", default="")
REALTIME_MODEL = os.getenv("OPENAI_REALTIME_MODEL", "gpt-4o-realtime-preview")
OPENAI_VOICE = os.getenv("OPENAI_VOICE", "verse")

ADMIN_API_KEY = os.getenv("ADMIN_API_KEY", "")

BASE_URL = os.getenv("BASE_URL", "").strip()
PUBLIC_BASE_URL = os.getenv("PUBLIC_BASE_URL", "").strip()

DATABASE_URL = os.getenv("DATABASE_URL") or "sqlite:///./dev.db"

# Exotel REST creds (optional for future features)
EXOTEL_SID = os.getenv("EXOTEL_SID", "")
EXOTEL_TOKEN = os.getenv("EXOTEL_TOKEN", "")
EXO_CALLER_ID = os.getenv("EXO_CALLER_ID", "")
EXOTEL_FLOW_URL = os.getenv("EXOTEL_FLOW_URL", "")
EXO_SUBDOMAIN = os.getenv("EXO_SUBDOMAIN", "api.in")


# ---------------- DB ----------------
class Base(DeclarativeBase):
    pass


class Tenant(Base):
    __tablename__ = "tenants"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(120))
    exotel_virtual_number: Mapped[str] = mapped_column(String(40), unique=True, index=True)
    system_prompt: Mapped[str] = mapped_column(Text, default="")
    transfer_number: Mapped[str] = mapped_column(String(40), default="")
    business_hours_json: Mapped[str] = mapped_column(Text, default="{}")
    created_at: Mapped[dt.datetime] = mapped_column(DateTime, default=dt.datetime.utcnow)

    calls: Mapped[list["Call"]] = relationship(back_populates="tenant")


class Call(Base):
    __tablename__ = "calls"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    tenant_id: Mapped[int] = mapped_column(ForeignKey("tenants.id"), index=True)
    stream_sid: Mapped[str] = mapped_column(String(120), index=True, default="")
    call_sid: Mapped[str] = mapped_column(String(120), index=True, default="")
    from_number: Mapped[str] = mapped_column(String(40), default="")
    to_number: Mapped[str] = mapped_column(String(40), default="")
    started_at: Mapped[dt.datetime] = mapped_column(DateTime, default=dt.datetime.utcnow)
    ended_at: Mapped[Optional[dt.datetime]] = mapped_column(DateTime, nullable=True)
    status: Mapped[str] = mapped_column(String(40), default="in_progress")
    transcript: Mapped[str] = mapped_column(Text, default="")

    tenant: Mapped["Tenant"] = relationship(back_populates="calls")


engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {},
)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)
Base.metadata.create_all(engine)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# ---------------- Audio helpers (no audioop; Python 3.13 safe) ----------------
# Exotel <-> OpenAI mapping:
#   Exotel: PCM16 mono @ 8000
#   OpenAI: PCM16 mono @ 24000
#
# We use a simple 3x upsample (repeat samples) and 3x downsample (take every 3rd).
# This is not audiophile-grade but works for speech and is fast and dependency-free.

def pcm16_8k_to_24k(pcm8: bytes) -> bytes:
    if not pcm8:
        return b""
    # int16 samples
    import array
    a = array.array("h")
    a.frombytes(pcm8)
    out = array.array("h")
    for s in a:
        out.append(s); out.append(s); out.append(s)
    return out.tobytes()

def pcm16_24k_to_8k(pcm24: bytes) -> bytes:
    if not pcm24:
        return b""
    import array
    a = array.array("h")
    a.frombytes(pcm24)
    out = array.array("h")
    out.extend(a[0::3])
    return out.tobytes()

def chunk_bytes(b: bytes, chunk_size: int) -> list[bytes]:
    return [b[i:i+chunk_size] for i in range(0, len(b), chunk_size)]


# ---------------- FastAPI ----------------
app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


@app.get("/health")
async def health():
    return PlainTextResponse("ok", status_code=200)


@app.get("/diag")
async def diag():
    return {
        "openai_key_present": bool(OPENAI_API_KEY),
        "admin_key_present": bool(ADMIN_API_KEY),
        "db_url": "sqlite" if DATABASE_URL.startswith("sqlite") else "non-sqlite",
        "public_base_url_set": bool(PUBLIC_BASE_URL or BASE_URL),
        "exotel_rest_creds_present": bool(EXOTEL_SID and EXOTEL_TOKEN),
    }


def _require_admin(x_admin_key: str = Header(default="")):
    if not ADMIN_API_KEY or x_admin_key != ADMIN_API_KEY:
        raise HTTPException(status_code=401, detail="Unauthorized")
    return True


# ---------------- Admin: Tenants ----------------
@app.get("/admin/tenants")
def list_tenants(_: bool = Depends(_require_admin), db: Session = Depends(get_db)):
    rows = db.query(Tenant).order_by(Tenant.id.desc()).all()
    return [
        {
            "id": t.id,
            "name": t.name,
            "exotel_virtual_number": t.exotel_virtual_number,
            "transfer_number": t.transfer_number,
            "created_at": t.created_at.isoformat(),
        }
        for t in rows
    ]


@app.post("/admin/tenants")
def create_tenant(payload: Dict[str, Any], _: bool = Depends(_require_admin), db: Session = Depends(get_db)):
    name = (payload.get("name") or "").strip()
    exo = (payload.get("exotel_virtual_number") or "").strip()
    system_prompt = (payload.get("system_prompt") or "").strip()
    transfer_number = (payload.get("transfer_number") or "").strip()
    if not name or not exo:
        raise HTTPException(status_code=400, detail="name and exotel_virtual_number required")
    t = Tenant(
        name=name,
        exotel_virtual_number=exo,
        system_prompt=system_prompt,
        transfer_number=transfer_number,
        business_hours_json=json.dumps(payload.get("business_hours") or {}),
    )
    db.add(t)
    db.commit()
    db.refresh(t)
    return {"id": t.id, "name": t.name, "exotel_virtual_number": t.exotel_virtual_number}


# ---------------- Exotel bootstrap ----------------
@app.get("/exotel/ws-bootstrap")
async def exotel_ws_bootstrap(request: Request, CallTo: str = ""):
    """
    Exotel Voicebot applet can call this and expects JSON: {"url": "wss://<host>/ws/exotel/<CallTo>"}
    """
    to_number = (CallTo or "").strip()
    # Determine base host
    base = (BASE_URL or PUBLIC_BASE_URL).strip()
    if base:
        base = base.replace("https://", "").replace("http://", "").rstrip("/")
    else:
        # derive from incoming host
        base = request.headers.get("x-forwarded-host") or request.headers.get("host") or ""
        base = base.split(",")[0].strip()
    url = f"wss://{base}/ws/exotel/{to_number}" if to_number else f"wss://{base}/ws/exotel"
    logger.info("Bootstrap served: %s", url)
    return {"url": url}


# ---------------- Core Exotel <-> OpenAI session ----------------
class OpenAIRealtime:
    def __init__(self, api_key: str, model: str, voice: str, instructions: str):
        self.api_key = api_key
        self.model = model
        self.voice = voice
        self.instructions = instructions
        self.session: Optional[aiohttp.ClientSession] = None
        self.ws: Optional[aiohttp.ClientWebSocketResponse] = None
        self.reader_task: Optional[asyncio.Task] = None
        self.on_audio: Optional[callable] = None  # callback(pcm24_bytes)
        self._closed = False

    async def connect(self):
        if not self.api_key:
            raise RuntimeError("OPENAI_API_KEY missing")
        headers = {"Authorization": f"Bearer {self.api_key}", "OpenAI-Beta": "realtime=v1"}
        url = f"wss://api.openai.com/v1/realtime?model={self.model}"
        self.session = aiohttp.ClientSession()
        self.ws = await self.session.ws_connect(url, headers=headers)

        # Session config
        await self.ws.send_json({
            "type": "session.update",
            "session": {
                "input_audio_format": "pcm16",
                "output_audio_format": "pcm16",
                "turn_detection": {
                    "type": "server_vad",
                    "threshold": 0.5,
                    "prefix_padding_ms": 200,
                    "silence_duration_ms": 600
                },
                "voice": self.voice,
                "instructions": self.instructions,
            }
        })

        # Start reader
        self.reader_task = asyncio.create_task(self._reader())

    async def _reader(self):
        assert self.ws is not None
        try:
            async for msg in self.ws:
                if msg.type == WSMsgType.TEXT:
                    evt = msg.json()
                    et = evt.get("type")

                    if et == "response.audio.delta":
                        b64 = evt.get("delta")
                        if b64 and self.on_audio:
                            pcm24 = base64.b64decode(b64)
                            self.on_audio(pcm24)

                    elif et in ("input_audio_buffer.speech_stopped", "input_audio_buffer.timeout_triggered"):
                        # Ensure model responds after user stops speaking
                        await self.ws.send_json({"type": "input_audio_buffer.commit"})
                        await self.ws.send_json({
                            "type": "response.create",
                            "response": {"modalities": ["audio", "text"], "instructions": "Reply briefly and helpfully."}
                        })

                    elif et == "error":
                        logger.error("OpenAI error: %s", evt)

                elif msg.type == WSMsgType.ERROR:
                    logger.error("OpenAI ws error")
                    break
        except Exception as e:
            logger.exception("OpenAI reader error: %s", e)

    async def send_audio_b64(self, b64_pcm16_24k: str):
        if not self.ws or self.ws.closed:
            return
        await self.ws.send_json({"type": "input_audio_buffer.append", "audio": b64_pcm16_24k})

    async def greet(self, text: str):
        if not self.ws or self.ws.closed:
            return
        await self.ws.send_json({
            "type": "response.create",
            "response": {"modalities": ["audio", "text"], "instructions": text}
        })

    async def close(self):
        if self._closed:
            return
        self._closed = True
        try:
            if self.reader_task and not self.reader_task.done():
                self.reader_task.cancel()
        except Exception:
            pass
        try:
            if self.ws and not self.ws.closed:
                await self.ws.close()
        except Exception:
            pass
        try:
            if self.session:
                await self.session.close()
        except Exception:
            pass


class ExotelVoiceSession:
    def __init__(self, ws: WebSocket, tenant: Tenant, db: Session):
        self.ws = ws
        self.tenant = tenant
        self.db = db

        self.stream_sid: Optional[str] = None
        self.call_sid: str = ""
        self.from_number: str = ""
        self.to_number: str = tenant.exotel_virtual_number

        self.ai: Optional[OpenAIRealtime] = None
        self.ai_started: bool = False

        # outbound audio buffering
        self._outbuf = bytearray()
        self._chunk_size = 3200  # 100ms at 8kHz PCM16 (recommended stable)

    async def _ensure_ai(self, reason: str):
        if self.ai_started:
            return
        self.ai_started = True
        prompt = (self.tenant.system_prompt or "").strip() or (
            "You are a helpful receptionist for a local business. "
            "Keep replies short and ask clarifying questions."
        )

        self.ai = OpenAIRealtime(
            api_key=OPENAI_API_KEY,
            model=REALTIME_MODEL,
            voice=OPENAI_VOICE,
            instructions=prompt,
        )

        # attach audio callback
        def on_audio(pcm24: bytes):
            pcm8 = pcm16_24k_to_8k(pcm24)
            self._outbuf.extend(pcm8)
            # schedule flush
            asyncio.create_task(self._flush_out())

        self.ai.on_audio = on_audio

        await self.ai.connect()
        logger.info("AI session started (%s)", reason)

        # Send greeting ASAP
        await self.ai.greet("Greet the caller politely and ask how you can help. Keep it short.")

    async def _flush_out(self):
        if not self.stream_sid:
            return
        while len(self._outbuf) >= self._chunk_size:
            chunk = bytes(self._outbuf[:self._chunk_size])
            del self._outbuf[:self._chunk_size]
            out_b64 = base64.b64encode(chunk).decode("ascii")
            msg = {"event": "media", "stream_sid": self.stream_sid, "media": {"payload": out_b64}}
            s = json.dumps(msg)
            logger.info("WS OUT event (first 200 chars): %s", s[:200])
            try:
                await self.ws.send_text(s)
            except Exception:
                break

    async def handle(self, evt: Dict[str, Any]):
        etype = (evt.get("event") or "").lower()

        if etype == "connected":
            await self._ensure_ai("connected")
            return

        if etype == "start":
            start_obj = evt.get("start") or {}
            self.stream_sid = start_obj.get("stream_sid") or start_obj.get("streamSid") or evt.get("stream_sid") or evt.get("streamSid")
            self.call_sid = start_obj.get("call_sid") or start_obj.get("callSid") or ""
            self.from_number = start_obj.get("from") or start_obj.get("CallFrom") or ""
            # upsert call row
            if self.stream_sid:
                existing = self.db.query(Call).filter(Call.stream_sid == self.stream_sid).first()
                if not existing:
                    c = Call(
                        tenant_id=self.tenant.id,
                        stream_sid=self.stream_sid,
                        call_sid=self.call_sid,
                        from_number=self.from_number,
                        to_number=self.to_number,
                        status="in_progress",
                    )
                    self.db.add(c)
                    self.db.commit()
            await self._ensure_ai("start")
            return

        if etype == "media":
            # Start-on-media failsafe
            if not self.stream_sid:
                self.stream_sid = evt.get("stream_sid") or evt.get("streamSid") or (evt.get("media") or {}).get("stream_sid")
            if not self.ai_started:
                await self._ensure_ai("media")

            media = evt.get("media") or {}
            payload_b64 = media.get("payload")
            if not payload_b64:
                return

            if not self.ai or not self.ai.ws or self.ai.ws.closed:
                return

            try:
                pcm8 = base64.b64decode(payload_b64)
            except Exception:
                return

            pcm24 = pcm16_8k_to_24k(pcm8)
            b64_24 = base64.b64encode(pcm24).decode("ascii")
            await self.ai.send_audio_b64(b64_24)
            return

        if etype == "stop":
            reason = (evt.get("stop") or {}).get("reason") or ""
            logger.info("Exotel stop sid=%s reason=%s", self.stream_sid, reason)
            # mark call ended
            if self.stream_sid:
                c = self.db.query(Call).filter(Call.stream_sid == self.stream_sid).first()
                if c:
                    c.status = "ended"
                    c.ended_at = dt.datetime.utcnow()
                    self.db.commit()
            return

    async def close(self):
        try:
            if self.ai:
                await self.ai.close()
        except Exception:
            pass


# ---------------- WebSocket route (tenant by called number) ----------------
@app.websocket("/ws/exotel/{to_number}")
async def exotel_ws_path(websocket: WebSocket, to_number: str, db: Session = Depends(get_db)):
    await websocket.accept()
    tenant = db.query(Tenant).filter(Tenant.exotel_virtual_number == (to_number or "")).first()
    logger.info("WS connected: to=%s tenant=%s", to_number, tenant.id if tenant else None)

    if not tenant:
        await websocket.close()
        return

    if not OPENAI_API_KEY:
        logger.error("No OPENAI_API_KEY; closing Exotel stream.")
        await websocket.close()
        return

    session = ExotelVoiceSession(websocket, tenant, db)

    try:
        while True:
            msg = await websocket.receive_text()
            logger.info("WS IN event (first 200 chars): %s", msg[:200])
            evt = json.loads(msg)
            await session.handle(evt)
    except WebSocketDisconnect:
        logger.info("Exotel WS disconnected")
    except Exception as e:
        logger.exception("Exotel WS error: %s", e)
    finally:
        await session.close()
        try:
            await websocket.close()
        except Exception:
            pass
