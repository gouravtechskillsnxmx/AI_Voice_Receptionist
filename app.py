import os, json, time, base64, asyncio, logging
from typing import Optional, Dict, Any
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
import aiohttp

# --- Local SaaS storage (SQLite today, Postgres later) ---

from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime
from sqlalchemy.orm import sessionmaker, declarative_base
import datetime as _dt


LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(level=getattr(logging, LOG_LEVEL, logging.INFO))
log = logging.getLogger("voicebridge")

app = FastAPI()


# -------------------- Local DB (Tenants + Call Events) --------------------
DATABASE_URL = (os.getenv("DATABASE_URL") or "").strip() or "sqlite:///./voicebridge.db"
_engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {},
    pool_pre_ping=True,
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=_engine)
Base = declarative_base()

class LocalTenant(Base):
    __tablename__ = "tenants"
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(200), nullable=False, default="Tenant")
    # "to_number" a.k.a Exotel virtual number mapped to this tenant
    to_number = Column(String(50), unique=True, index=True, nullable=False)
    system_prompt = Column(Text, nullable=False, default="You are a helpful receptionist. Keep replies short.")
    transfer_number = Column(String(50), nullable=True)
    missed_call_sms_to = Column(String(50), nullable=True)
    created_at = Column(DateTime, default=_dt.datetime.utcnow)

class CallEvent(Base):
    __tablename__ = "call_events"
    id = Column(Integer, primary_key=True)
    tenant_id = Column(Integer, index=True, nullable=True)
    provider = Column(String(32), default="exotel")
    call_sid = Column(String(128), index=True, nullable=False)
    stream_sid = Column(String(128), index=True, nullable=True)
    event_type = Column(String(64), index=True, nullable=False)
    payload_json = Column(Text, default="{}")
    created_at = Column(DateTime, default=_dt.datetime.utcnow)

def _db_init():
    Base.metadata.create_all(bind=_engine)

_db_init()

def _db():
    db = SessionLocal()
    try:
        return db
    except Exception:
        db.close()
        raise


PUBLIC_BASE_URL = (os.getenv("PUBLIC_BASE_URL") or "").strip()
SAAS_BASE_URL = (os.getenv("SAAS_BASE_URL") or "").strip().rstrip("/")
VOICE_WEBHOOK_SECRET = (os.getenv("VOICE_WEBHOOK_SECRET") or "").strip()
OPENAI_API_KEY = (os.getenv("OPENAI_API_KEY") or os.getenv("OpenAI_Key") or "").strip()

OPENAI_REALTIME_MODEL = os.getenv("OPENAI_REALTIME_MODEL", "gpt-4o-realtime-preview")

def _host_only(v: str) -> str:
    return v.replace("https://","").replace("http://","").strip().strip("/")

async def _saas_get_tenant(to_number: str) -> Dict[str, Any]:
    # If SAAS_BASE_URL is not provided, we use the local SQLite tenant table (single-service mode).
    if not SAAS_BASE_URL:
        db = _db()
        try:
            t = db.query(LocalTenant).filter(LocalTenant.to_number == (to_number or "").strip()).first()
            if not t:
                return {"tenant_id": None, "system_prompt": "You are a helpful receptionist. Keep replies short.",
                        "transfer_number": None, "missed_call_sms_to": None}
            return {"tenant_id": t.id, "system_prompt": t.system_prompt,
                    "transfer_number": t.transfer_number, "missed_call_sms_to": t.missed_call_sms_to}
        except Exception as e:
            log.warning("Local tenant lookup error: %r", e)
            return {"tenant_id": None, "system_prompt": "You are a helpful receptionist. Keep replies short.",
                    "transfer_number": None, "missed_call_sms_to": None}
        finally:
            try: db.close()
            except Exception: pass
    try:
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=8)) as s:
            async with s.get(
                f"{SAAS_BASE_URL}/internal/tenant-by-number",
                params={"to": to_number},
                headers={"X-Voice-Secret": VOICE_WEBHOOK_SECRET},
            ) as r:
                if r.status != 200:
                    txt = await r.text()
                    log.warning("Tenant lookup failed status=%s body=%s", r.status, txt[:200])
                    return {"tenant_id": None, "system_prompt": "You are a helpful receptionist. Keep replies short.",
                            "transfer_number": None, "missed_call_sms_to": None}
                return await r.json()
    except Exception as e:
        log.warning("Tenant lookup error: %r", e)
        return {"tenant_id": None, "system_prompt": "You are a helpful receptionist. Keep replies short.",
                "transfer_number": None, "missed_call_sms_to": None}

async def _saas_post_event(event: Dict[str, Any]) -> None:
    # Single-service mode: store events locally when SAAS_BASE_URL is not set.
    if not SAAS_BASE_URL:
        db = _db()
        try:
            tenant_id = event.get("tenant_id")
            db.add(CallEvent(
                tenant_id=int(tenant_id) if tenant_id is not None else None,
                provider=(event.get("provider") or "exotel"),
                call_sid=(event.get("call_sid") or "unknown"),
                stream_sid=(event.get("stream_sid") or None),
                event_type=(event.get("type") or "event"),
                payload_json=json.dumps(event.get("payload") or {})[:50000],
            ))
            db.commit()
        except Exception as e:
            log.warning("Local event store failed: %r", e)
        finally:
            try: db.close()
            except Exception: pass
        return
    try:
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=8)) as s:
            async with s.post(
                f"{SAAS_BASE_URL}/webhooks/voicebridge",
                json=event,
                headers={"X-Voice-Secret": VOICE_WEBHOOK_SECRET, "X-VoiceBridge-Version":"1"},
            ) as r:
                if r.status >= 300:
                    txt = await r.text()
                    log.warning("Webhook post failed status=%s body=%s", r.status, txt[:200])
    except Exception as e:
        log.warning("Webhook post error: %r", e)

@app.get("/health")
async def health():
    return {"ok": True}



# -------------------- Admin API (create/list tenants) --------------------
ADMIN_API_KEY = (os.getenv("ADMIN_API_KEY") or os.getenv("Admin_API_Key") or "").strip()

def _require_admin(x_admin_key: str | None):
    if not ADMIN_API_KEY or not x_admin_key or x_admin_key.strip() != ADMIN_API_KEY:
        from fastapi import HTTPException
        raise HTTPException(status_code=401, detail="Unauthorized")

@app.post("/admin/tenants")
async def admin_create_tenant(payload: dict, request: Request):
    x_admin_key = request.headers.get("X-Admin-Key") or request.headers.get("x-admin-key")
    _require_admin(x_admin_key)
    name = (payload.get("name") or "").strip() or "Tenant"
    to_number = (payload.get("to_number") or payload.get("exotel_virtual_number") or "").strip()
    system_prompt = (payload.get("system_prompt") or "").strip() or "You are a helpful receptionist. Keep replies short."
    transfer_number = (payload.get("transfer_number") or payload.get("forward_to_number") or "").strip() or None
    missed_call_sms_to = (payload.get("missed_call_sms_to") or "").strip() or None
    if not to_number:
        from fastapi import HTTPException
        raise HTTPException(status_code=400, detail="to_number (or exotel_virtual_number) required")

    db = _db()
    try:
        # upsert by to_number
        t = db.query(LocalTenant).filter(LocalTenant.to_number == to_number).first()
        if t is None:
            t = LocalTenant(
                name=name,
                to_number=to_number,
                system_prompt=system_prompt,
                transfer_number=transfer_number,
                missed_call_sms_to=missed_call_sms_to,
            )
            db.add(t)
            db.commit()
            db.refresh(t)
        else:
            t.name = name
            t.system_prompt = system_prompt
            t.transfer_number = transfer_number
            t.missed_call_sms_to = missed_call_sms_to
            db.add(t)
            db.commit()
            db.refresh(t)
        return {
            "tenant_id": t.id,
            "name": t.name,
            "to_number": t.to_number,
            "system_prompt": t.system_prompt,
            "transfer_number": t.transfer_number,
            "missed_call_sms_to": t.missed_call_sms_to,
        }
    finally:
        db.close()

@app.get("/admin/tenants")
async def admin_list_tenants(request: Request):
    x_admin_key = request.headers.get("X-Admin-Key") or request.headers.get("x-admin-key")
    _require_admin(x_admin_key)
    db = _db()
    try:
        rows = db.query(LocalTenant).order_by(LocalTenant.id.desc()).all()
        return [
            {
                "tenant_id": r.id,
                "name": r.name,
                "to_number": r.to_number,
                "system_prompt": r.system_prompt,
                "transfer_number": r.transfer_number,
                "missed_call_sms_to": r.missed_call_sms_to,
            }
            for r in rows
        ]
    finally:
        db.close()

# -------------------- Internal endpoints for adapter mode --------------------
@app.get("/internal/tenant-by-number")
async def internal_tenant_by_number(to: str = "", request: Request = None):
    # Optional shared secret
    if VOICE_WEBHOOK_SECRET:
        got = (request.headers.get("X-Voice-Secret") if request else "") or ""
        if got.strip() != VOICE_WEBHOOK_SECRET.strip():
            from fastapi import HTTPException
            raise HTTPException(status_code=401, detail="Unauthorized")

    to_number = (to or "").strip()
    db = _db()
    try:
        t = db.query(LocalTenant).filter(LocalTenant.to_number == to_number).first()
        if not t:
            from fastapi import HTTPException
            raise HTTPException(status_code=404, detail="Tenant not found")
        return {
            "tenant_id": t.id,
            "system_prompt": t.system_prompt,
            "transfer_number": t.transfer_number,
            "missed_call_sms_to": t.missed_call_sms_to,
        }
    finally:
        db.close()

@app.post("/webhooks/voicebridge")
async def webhook_voicebridge(event: dict, request: Request):
    if VOICE_WEBHOOK_SECRET:
        got = (request.headers.get("X-Voice-Secret") or "").strip()
        if got != VOICE_WEBHOOK_SECRET.strip():
            from fastapi import HTTPException
            raise HTTPException(status_code=401, detail="Unauthorized")

    db = _db()
    try:
        tenant_id = event.get("tenant_id")
        call_sid = (event.get("call_sid") or "").strip() or "unknown"
        stream_sid = (event.get("stream_sid") or "").strip() or None
        etype = (event.get("type") or "event").strip()
        payload = event.get("payload") or {}
        db.add(CallEvent(
            tenant_id=int(tenant_id) if tenant_id is not None else None,
            provider=(event.get("provider") or "exotel"),
            call_sid=call_sid,
            stream_sid=stream_sid,
            event_type=etype,
            payload_json=json.dumps(payload)[:50000],
        ))
        db.commit()
        return {"ok": True}
    finally:
        db.close()


@app.get("/exotel/ws-bootstrap")
async def exotel_ws_bootstrap(request: Request):
    q = dict(request.query_params)
    call_to = q.get("CallTo") or q.get("To") or q.get("to") or ""
    call_sid = q.get("CallSid") or q.get("call_sid") or "test"
    host = _host_only(PUBLIC_BASE_URL) or _host_only(request.headers.get("host",""))
    url = f"wss://{host}/exotel-media?to={call_to}&call_sid={call_sid}"
    log.info("Bootstrap served: %s", url)
    return {"url": url}

class BridgeSession:
    def __init__(self, ws: WebSocket, to_number: str, call_sid: str):
        self.ws = ws
        self.to_number = to_number
        self.call_sid = call_sid
        self.stream_sid: Optional[str] = None
        self.seq = 1
        self.chunk = 1
        self.first_out_logged = False
        self.started_at = time.time()
        self.sent_any_audio = False

        self.tenant_id: Optional[int] = None
        self.system_prompt: str = "You are a helpful receptionist. Keep replies short."
        self.transfer_number: Optional[str] = None
        self.missed_call_sms_to: Optional[str] = None

        self.sr = 8000
        self._ai_started = False
        self._aiohttp: Optional[aiohttp.ClientSession] = None
        self._ai_ws: Optional[aiohttp.ClientWebSocketResponse] = None

    async def start_ai(self):
        if self._ai_started:
            return
        self._ai_started = True

        cfg = await _saas_get_tenant(self.to_number)
        self.tenant_id = cfg.get("tenant_id")
        self.system_prompt = cfg.get("system_prompt") or self.system_prompt
        self.transfer_number = cfg.get("transfer_number")
        self.missed_call_sms_to = cfg.get("missed_call_sms_to")

        await _saas_post_event(self._event("call.started", {"sample_rate": self.sr}))

        if not OPENAI_API_KEY:
            log.error("OPENAI_API_KEY missing — cannot start realtime")
            return

        self._aiohttp = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=None))
        headers = {"Authorization": f"Bearer {OPENAI_API_KEY}", "OpenAI-Beta": "realtime=v1"}
        self._ai_ws = await self._aiohttp.ws_connect(
            f"wss://api.openai.com/v1/realtime?model={OPENAI_REALTIME_MODEL}",
            headers=headers,
            heartbeat=20,
        )

        await self._ai_ws.send_json({
            "type":"session.update",
            "session":{
                "modalities":["audio","text"],
                "instructions": self.system_prompt,
                "input_audio_format":{"type":"pcm16","sample_rate": self.sr},
                "output_audio_format":{"type":"pcm16","sample_rate": self.sr},
                "turn_detection":{"type":"server_vad"}
            }
        })

        # Proactively speak first so the call doesn't feel "dead" and Exotel doesn't time out.
        try:
            await self._ai_ws.send_json({
                "type": "response.create",
                "response": {
                    "instructions": "Greet the caller in one short sentence and ask how you can help. Keep it brief.",
                    "modalities": ["audio", "text"]
                }
            })
        except Exception as e:
            log.warning("Failed to trigger initial greeting: %r", e)

        asyncio.create_task(self._pump_openai_to_exotel())
        log.info("AI session started tenant_id=%s to=%s", self.tenant_id, self.to_number)

    def _event(self, etype: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "event_id": f"evt_{int(time.time()*1000)}_{self.call_sid}",
            "type": etype,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "tenant_id": self.tenant_id,
            "provider": "exotel",
            "call_sid": self.call_sid,
            "stream_sid": self.stream_sid,
            "from_number": None,
            "to_number": self.to_number,
            "payload": payload,
        }

    async def close(self, end_reason: str = "stop"):
        duration = int(time.time() - self.started_at)
        missed = (duration < 8) or (not self.sent_any_audio)
        await _saas_post_event(self._event("call.ended", {"duration_sec": duration, "end_reason": end_reason, "missed_call": missed}))
        try:
            if self._ai_ws:
                await self._ai_ws.close()
        except Exception:
            pass
        try:
            if self._aiohttp:
                await self._aiohttp.close()
        except Exception:
            pass

    async def _send_exotel_media(self, pcm16_bytes: bytes):
        if not self.stream_sid:
            return
        payload = base64.b64encode(pcm16_bytes).decode("ascii")
        msg = {
            "event": "media",
            "stream_sid": self.stream_sid,
            "sequence_number": str(self.seq),
            "media": {
                "chunk": str(self.chunk),
                "timestamp": str(int(time.time()*1000)),
                "payload": payload,
            }
        }
        await self.ws.send_text(json.dumps(msg))
        self.seq += 1
        self.chunk += 1
        self.sent_any_audio = True
        if not self.first_out_logged:
            self.first_out_logged = True
            log.info("WS OUT media stream_sid=%s bytes=%d", self.stream_sid, len(pcm16_bytes))

    async def _pump_openai_to_exotel(self):
        assert self._ai_ws is not None
        async for m in self._ai_ws:
            if m.type == aiohttp.WSMsgType.TEXT:
                evt = json.loads(m.data)
                t = evt.get("type")
                if t == "response.output_audio.delta":
                    delta = evt.get("delta") or ""
                    if delta:
                        await self._send_exotel_media(base64.b64decode(delta))
            elif m.type in (aiohttp.WSMsgType.CLOSE, aiohttp.WSMsgType.CLOSED):
                break

    async def handle_exotel(self, evt: Dict[str, Any]) -> Optional[str]:
        et = evt.get("event")
        try:
            log.info("WS IN event=%s first200=%s", et, json.dumps(evt)[:200])
        except Exception:
            pass
        if et == "connected":
            # Exotel usually sends connected -> start. Start contains to/call_sid/stream_sid.
            # If we don't yet know the destination number, wait for start or media.
            if self.to_number and (not self._ai_started):
                await self.start_ai()
        elif et == "start":
            s = evt.get("start") or {}
            self.stream_sid = evt.get("stream_sid") or s.get("stream_sid") or self.stream_sid
            # Capture call/to from start if query params were not passed
            if not self.call_sid or self.call_sid == "unknown":
                self.call_sid = s.get("call_sid") or evt.get("call_sid") or self.call_sid
            if not self.to_number:
                self.to_number = s.get("to") or s.get("CallTo") or s.get("call_to") or self.to_number
            if not self._ai_started:
                await self.start_ai()
        elif et == "media":
            self.stream_sid = evt.get("stream_sid") or self.stream_sid
            media = evt.get("media") or {}
            payload = media.get("payload") or ""
            if (not self._ai_started):
                await self.start_ai()
            if payload and self._ai_ws:
                await self._ai_ws.send_json({"type":"input_audio_buffer.append","audio": payload})
        elif et == "stop":
            return "stop"
        return None

@app.websocket("/exotel-media")
async def exotel_media(ws: WebSocket):
    await ws.accept()
    to_number = (ws.query_params.get("to") or "").strip()
    call_sid = (ws.query_params.get("call_sid") or "unknown").strip()
    log.info("Exotel WS connected to=%s call_sid=%s qs=%s", to_number, call_sid, dict(ws.query_params))

    sess = BridgeSession(ws, to_number, call_sid)
    try:
        while True:
            evt = json.loads(await ws.receive_text())
            r = await sess.handle_exotel(evt)
            if r == "stop":
                break
    except WebSocketDisconnect:
        await sess.close("disconnect")
    except Exception as e:
        log.exception("WS error: %r", e)
        await sess.close("error")
    finally:
        try:
            await ws.close()
        except Exception:
            pass
