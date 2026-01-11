
# LLM Server (Ollama + NVIDIA NIM) com Gateway OpenAI-like

Servidor local de LLM com uma API compatível com OpenAI:
- `GET /v1/models`
- `POST /v1/chat/completions`

Backends suportados:
- **Ollama** (padrão) + download automático de modelos via `.env` (serviço `ollama-init`)
- **NVIDIA NIM** (opcional) via `--profile nim`

> **Linux only (do jeito que está):** este compose usa `network_mode: host`.

## Portas

- Gateway: `http://127.0.0.1:8080`
- Ollama (quando profile `ollama`): `http://127.0.0.1:11434`
- NIM (quando profile `nim`): `http://127.0.0.1:8000`

---

## Requisitos

- Docker + Docker Compose
- Internet para o container baixar modelos (Ollama/NIM)
- Para **NIM**: GPU NVIDIA + drivers + runtime NVIDIA no Docker

---

## Configuração (.env)

Exemplo mínimo usando **Ollama**:

```env
API_KEY=dev123
LLM_PROVIDER=ollama
OLLAMA_BASE_URL=http://127.0.0.1:11434
DEFAULT_MODEL=deepseek-r1:8b

# Modelos que o init vai baixar automaticamente
OLLAMA_MODELS=deepseek-r1:8b


Exemplo mínimo para **NIM** (opcional):

NGC_API_KEY=SEU_TOKEN_NGC
LLM_PROVIDER=nim
NIM_BASE_URL=http://127.0.0.1:8000/v1
```

---

## Subir com Ollama (baixa modelo automaticamente)

```bash
make up
```

Acompanhar o download do modelo (feito pelo `ollama-init`):

```bash
docker compose logs -f ollama-init
```

---

## Subir com NVIDIA NIM (opcional)

> **Sim, ainda dá pra usar NIM**: ele está no `docker-compose.yml` e sobe via `make up-nim`.
> O problema é que pode exigir mais GPU/VRAM e fazer downloads/caches grandes.

```bash
make up-nim
```

---

## Testar o servidor (gateway)

Health:

```bash
curl -s http://127.0.0.1:8080/health | python -m json.tool
```

Listar modelos:

```bash
export API_KEY=dev123
curl -s http://127.0.0.1:8080/v1/models \
  -H "Authorization: Bearer $API_KEY" | python -m json.tool
```

Chat (OpenAI-like):

```bash
curl -s http://127.0.0.1:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $API_KEY" \
  -d '{
    "model": "deepseek-r1:8b",
    "messages": [{"role":"user","content":"Me diga um haicai sobre GPUs."}],
    "max_tokens": 80
  }' | python -m json.tool
```

---

## Smoke test

```bash
make smoke
```

---

## Parar tudo

```bash
make down
```

```
::contentReference[oaicite:0]{index=0}
```
