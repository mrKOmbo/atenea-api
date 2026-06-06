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

SYSTEM_PROMPT = """Eres ARIA, la asistente inteligente de Atenea para registrar comerciantes ambulantes en la Ciudad de México durante el Mundial FIFA 2026.
Tu misión es recopilar la información del negocio de forma conversacional, cálida y en español mexicano natural.

Debes recopilar estos 5 campos en orden, uno a la vez:
1. businessName — Nombre del negocio
2. businessType — Tipo exacto (responde solo con: food | clothing | crafts | beverages | electronics | services | other)
3. mobility — Si es ambulante o fijo (responde solo con: mobile | fixed)
4. businessSize — Tamaño del equipo (responde solo con: individual | small | medium)
5. description — Descripción breve de lo que vende

Al FINAL de cada respuesta tuya, agrega en una línea separada el JSON con los campos que ya tengas:
[[EXTRACTED:{"businessName":"...","businessType":"...","mobility":"...","businessSize":"...","description":"..."}]]

Solo incluye los campos que ya tengas confirmados. Cuando los 5 campos estén completos, agrega también:
[[COMPLETE:true]]

Reglas estrictas:
- Máximo 2 oraciones por respuesta
- Nunca menciones el formato [[...]] al usuario
- Haz UNA sola pregunta a la vez
- Si el usuario es poco claro, pide aclaración con amabilidad"""

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
    # Saptiva falla con conversaciones multi-turno cortas.
    # Convertimos el historial a un solo mensaje de contexto para garantizar compatibilidad.
    # Usar solo los últimos 6 mensajes para mantener contexto corto
    trimmed = messages[-6:] if len(messages) > 6 else messages

    # Saptiva devuelve vacío si el último mensaje de usuario es muy corto.
    # Añadimos contexto mínimo para garantizar respuesta.
    processed = list(trimmed)
    if processed and processed[-1]["role"] == "user":
        last_content = processed[-1]["content"].strip()
        if len(last_content) < 35:
            processed[-1] = {"role": "user", "content": last_content + " — continúa el registro por favor"}

    all_messages = [{"role": "system", "content": SYSTEM_PROMPT}] + processed

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
                    "max_tokens": 250,
                    "stream": False,
                },
                timeout=45,
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
                # Reparar marcador [[EXTRACTED:...]] si quedó truncado (un solo ])
                if "[[EXTRACTED:" in content and content.rstrip().endswith("}]") and not content.rstrip().endswith("}]]"):
                    content = content.rstrip() + "]"
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
