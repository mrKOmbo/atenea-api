"""
Servidor ligero para Atenea iOS — onboarding conversacional de comerciantes.
Usa Saptiva KAL. No requiere LangGraph ni Ollama.
Ejecutar: python atenea_server.py
"""
import os
import time
import requests
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List

# Cargar .env.local
for path in [".env.local", ".env"]:
    if os.path.exists(path):
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    os.environ.setdefault(k.strip(), v.strip())

SAPTIVA_KEY = os.environ.get("SAPTIVA_API_KEY", "")
SAPTIVA_URL = os.environ.get("SAPTIVA_BASE_URL", "https://api.saptiva.com")

SYSTEM_PROMPT = """Eres ARIA, asistente de Atenea. Registra comerciantes del Mundial FIFA 2026 CDMX.
Recopila 5 campos en orden, uno a la vez:
1. businessName 2. businessType (food|clothing|crafts|beverages|electronics|services|other)
3. mobility (mobile|fixed) 4. businessSize (individual|small|medium) 5. description

Al final de CADA respuesta agrega en línea aparte:
[[EXTRACTED:{"businessName":"...","businessType":"...","mobility":"...","businessSize":"...","description":"..."}]]
Solo campos confirmados. Cuando los 5 estén completos agrega también: [[COMPLETE:true]]

Reglas: máx 2 oraciones, UNA pregunta a la vez, español mexicano natural, nunca menciones [[...]]"""

app = FastAPI(title="Atenea Onboarding API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class Message(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    messages: List[Message]


def call_saptiva(messages: list, retries: int = 2) -> str:
    all_messages = [{"role": "system", "content": SYSTEM_PROMPT}] + messages

    for attempt in range(retries):
        try:
            resp = requests.post(
                f"{SAPTIVA_URL}/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {SAPTIVA_KEY}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": "Saptiva KAL",
                    "messages": all_messages,
                    "temperature": 0.7,
                    "max_tokens": 80,
                    "stream": False,
                },
                timeout=40,
            )

            print(f"[Saptiva] status={resp.status_code} attempt={attempt+1}")

            if not resp.ok:
                print(f"[Saptiva] error body: {resp.text[:200]}")
                if attempt < retries - 1:
                    time.sleep(2)
                    continue
                raise HTTPException(status_code=resp.status_code, detail=resp.text)

            data = resp.json()
            content = data["choices"][0]["message"]["content"]
            print(f"[Saptiva] content length={len(content)}")

            if content.strip():
                return content

            # Contenido vacío — reintentar
            print(f"[Saptiva] content vacío, reintentando...")
            if attempt < retries - 1:
                time.sleep(2)

        except requests.Timeout:
            print(f"[Saptiva] timeout en intento {attempt+1}")
            if attempt < retries - 1:
                time.sleep(2)
            else:
                raise HTTPException(status_code=504, detail="Saptiva no respondió a tiempo")

    raise HTTPException(status_code=502, detail="Saptiva no generó respuesta")


@app.post("/api/chat")
def chat(req: ChatRequest):
    if not SAPTIVA_KEY:
        raise HTTPException(status_code=500, detail="SAPTIVA_API_KEY no configurada")

    messages = [{"role": m.role, "content": m.content} for m in req.messages]
    content = call_saptiva(messages)
    return {"content": content}


@app.get("/health")
def health():
    return {"status": "ok", "saptiva_key_set": bool(SAPTIVA_KEY)}


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
