import os, json, logging
from datetime import datetime
from typing import Optional
from fastapi import FastAPI, HTTPException, Header, Request
from pydantic import BaseModel
from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime
from sqlalchemy.orm import sessionmaker, declarative_base

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(level=getattr(logging, LOG_LEVEL, logging.INFO))
log = logging.getLogger("saas")

app = FastAPI()

ADMIN_API_KEY = os.getenv("ADMIN_API_KEY", os.getenv("Admin_API_Key", ""))
VOICE_WEBHOOK_SECRET = os.getenv("VOICE_WEBHOOK_SECRET", "")
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./saas.db")

connect_args = {"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {}
engine = create_engine(DATABASE_URL, connect_args=connect_args, pool_pre_ping=True)
SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)
Base = declarative_base()

class Tenant(Base):
    __tablename__ = "tenants"
    id = Column(Integer, primary_key=True)
    name = Column(String(200), nullable=False)
    to_number = Column(String(50), unique=True, nullable=False)
    system_prompt = Column(Text, nullable=False)
    transfer_number = Column(String(50))
    missed_call_sms_to = Column(String(50))
    created_at = Column(DateTime, default=datetime.utcnow)

class VoiceEvent(Base):
    __tablename__ = "voice_events"
    id = Column(Integer, primary_key=True)
    event_id = Column(String(200))
    type = Column(String(80))
    tenant_id = Column(Integer)
    provider = Column(String(50))
    call_sid = Column(String(120))
    stream_sid = Column(String(120))
    to_number = Column(String(50))
    payload_json = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)

Base.metadata.create_all(bind=engine)

def require_admin(key: Optional[str]):
    if not ADMIN_API_KEY or key != ADMIN_API_KEY:
        raise HTTPException(status_code=401, detail="Unauthorized")

def require_voice_secret(secret: Optional[str]):
    if VOICE_WEBHOOK_SECRET and secret != VOICE_WEBHOOK_SECRET:
        raise HTTPException(status_code=401, detail="Unauthorized")

class TenantCreate(BaseModel):
    name: str
    to_number: str
    system_prompt: str
    transfer_number: Optional[str] = None
    missed_call_sms_to: Optional[str] = None

@app.get("/health")
async def health():
    return {"ok": True}

@app.post("/admin/tenants")
async def create_tenant(payload: TenantCreate, x_admin_key: Optional[str] = Header(None, alias="X-Admin-Key")):
    require_admin(x_admin_key)
    db = SessionLocal()
    try:
        t = Tenant(**payload.model_dump())
        db.add(t); db.commit(); db.refresh(t)
        return {"id": t.id, "name": t.name, "to_number": t.to_number}
    finally:
        db.close()

@app.get("/internal/tenant-by-number")
async def tenant_by_number(to: str, x_voice_secret: Optional[str] = Header(None, alias="X-Voice-Secret")):
    require_voice_secret(x_voice_secret)
    db = SessionLocal()
    try:
        t = db.query(Tenant).filter(Tenant.to_number == to).first()
        if not t:
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
async def voicebridge_webhook(request: Request, x_voice_secret: Optional[str] = Header(None, alias="X-Voice-Secret")):
    require_voice_secret(x_voice_secret)
    evt = await request.json()
    db = SessionLocal()
    try:
        row = VoiceEvent(
            event_id=evt.get("event_id"),
            type=evt.get("type"),
            tenant_id=evt.get("tenant_id"),
            provider=evt.get("provider"),
            call_sid=evt.get("call_sid"),
            stream_sid=evt.get("stream_sid"),
            to_number=evt.get("to_number"),
            payload_json=json.dumps(evt.get("payload") or {}),
        )
        db.add(row); db.commit()
    finally:
        db.close()
    log.info("Event received type=%s tenant=%s call_sid=%s", evt.get("type"), evt.get("tenant_id"), evt.get("call_sid"))
    return {"ok": True}
