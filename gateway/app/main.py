from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import JSONResponse
import httpx
import os
import time

app = FastAPI()

API_KEY = os.environ.get("API_KEY", "")

LLM_PROVIDER = os.environ.get("LLM_PROVIDER", "ollama").lower().strip()
NIM_BASE_URL = os.environ.get("NIM_BASE_URL", "http://nim:8000/v1").rstrip("/")
OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://127.0.0.1:11434").rstrip("/")
DEFAULT_MODEL = os.environ.get("DEFAULT_MODEL", "deepseek-r1:8b")

TIMEOUT = float(os.environ.get("LLM_TIMEOUT", "120.0"))

@app.get("/health")
async def health():
    # health do gateway
    return {"status": "ok", "provider": LLM_PROVIDER}

async def _auth_guard(authorization: str | None):
    if API_KEY:
        if not authorization or authorization != f"Bearer {API_KEY}":
            raise HTTPException(status_code=401, detail="unauthorized")

def _get_model(payload: dict) -> str:
    # se o client mandar model, usa; senão default
    model = payload.get("model") or DEFAULT_MODEL
    return model

@app.get("/v1/models")
async def list_models(authorization: str | None = Header(default=None)):
    await _auth_guard(authorization)

    try:
        async with httpx.AsyncClient(timeout=TIMEOUT) as client:
            if LLM_PROVIDER == "nim":
                r = await client.get(f"{NIM_BASE_URL}/models")
                return JSONResponse(status_code=r.status_code, content=r.json())
            elif LLM_PROVIDER == "ollama":
                # Ollama: /api/tags
                r = await client.get(f"{OLLAMA_BASE_URL}/api/tags")
                data = r.json()
                # Converter para formato OpenAI-like
                models = []
                for m in data.get("models", []):
                    models.append({
                        "id": m.get("name"),
                        "object": "model",
                        "created": 0,
                        "owned_by": "ollama",
                    })
                return {"object": "list", "data": models}
            else:
                return JSONResponse(status_code=400, content={"error": {"message": f"Unknown provider: {LLM_PROVIDER}"}})
    except httpx.RequestError as e:
        return JSONResponse(status_code=503, content={"error": {"message": f"Provider request failed: {str(e)}"}})

@app.post("/v1/chat/completions")
async def chat_completions(payload: dict, authorization: str | None = Header(default=None)):
    await _auth_guard(authorization)

    model = _get_model(payload)
    messages = payload.get("messages", [])
    max_tokens = payload.get("max_tokens")

    try:
        async with httpx.AsyncClient(timeout=TIMEOUT) as client:
            if LLM_PROVIDER == "nim":
                # Proxy direto estilo OpenAI
                r = await client.post(f"{NIM_BASE_URL}/chat/completions", json=payload)
                try:
                    content = r.json()
                except Exception:
                    content = {"error": {"message": r.text}}
                return JSONResponse(status_code=r.status_code, content=content)

            elif LLM_PROVIDER == "ollama":
                # Ollama native API: /api/chat
                ollama_payload = {
                    "model": model,
                    "messages": messages,
                    "stream": False,
                }

                # Opcional: mapear max_tokens -> num_predict
                options = {}
                if isinstance(max_tokens, int):
                    options["num_predict"] = max_tokens
                if options:
                    ollama_payload["options"] = options

                r = await client.post(f"{OLLAMA_BASE_URL}/api/chat", json=ollama_payload)
                data = r.json()

                # Converter resposta para OpenAI chat/completions
                # Ollama retorna {"message": {"role":"assistant","content":"..."}, ...}
                assistant_msg = (data.get("message") or {}).get("content", "")
                now = int(time.time())

                openai_like = {
                    "id": f"chatcmpl-ollama-{now}",
                    "object": "chat.completion",
                    "created": now,
                    "model": model,
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": assistant_msg},
                            "finish_reason": "stop",
                        }
                    ],
                }

                # (Opcional) usage aproximado se existir
                # Ollama às vezes tem "prompt_eval_count" e "eval_count"
                prompt_tokens = data.get("prompt_eval_count")
                completion_tokens = data.get("eval_count")
                if isinstance(prompt_tokens, int) and isinstance(completion_tokens, int):
                    openai_like["usage"] = {
                        "prompt_tokens": prompt_tokens,
                        "completion_tokens": completion_tokens,
                        "total_tokens": prompt_tokens + completion_tokens,
                    }

                return JSONResponse(status_code=r.status_code, content=openai_like)

            else:
                return JSONResponse(status_code=400, content={"error": {"message": f"Unknown provider: {LLM_PROVIDER}"}})

    except httpx.RequestError as e:
        return JSONResponse(status_code=503, content={"error": {"message": f"Provider request failed: {type(e).__name__}: {e!r}"}})
