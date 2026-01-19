import os
import httpx
import streamlit as st

GATEWAY_URL = os.getenv("GATEWAY_URL", "http://127.0.0.1:8080").rstrip("/")
API_KEY = os.getenv("API_KEY", "")
TIMEOUT = float(os.getenv("TIMEOUT", "120"))

st.set_page_config(page_title="Chat LLM", page_icon="💬")

def headers():
    h = {"Content-Type": "application/json"}
    if API_KEY:
        h["Authorization"] = f"Bearer {API_KEY}"
    return h

def list_models():
    with httpx.Client(timeout=TIMEOUT) as client:
        r = client.get(f"{GATEWAY_URL}/v1/models", headers=headers())
        r.raise_for_status()
        data = r.json()
        return [m["id"] for m in data.get("data", [])]

def chat(model, messages, max_tokens):
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
    }
    with httpx.Client(timeout=TIMEOUT) as client:
        r = client.post(f"{GATEWAY_URL}/v1/chat/completions", headers=headers(), json=payload)
        r.raise_for_status()
        data = r.json()
        return data["choices"][0]["message"]["content"]

st.title("💬 Chat com sua LLM")

with st.sidebar:
    st.subheader("Configuração")

    try:
        models = list_models()
    except Exception as e:
        st.error(f"Erro ao listar modelos: {e}")
        models = [os.getenv("DEFAULT_MODEL", "deepseek-r1:8b")]

    model = st.selectbox("Modelo", models)
    max_tokens = st.slider("max_tokens", 32, 1024, 256)

    if st.button("Limpar conversa"):
        st.session_state.messages = [
            {"role": "system", "content": "Você é um assistente útil e direto."}
        ]

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "system", "content": "Você é um assistente útil e direto."}
    ]

for msg in st.session_state.messages:
    if msg["role"] == "system":
        continue
    with st.chat_message("user" if msg["role"] == "user" else "assistant"):
        st.write(msg["content"])

prompt = st.chat_input("Digite sua mensagem")
if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.write(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Pensando..."):
            try:
                answer = chat(model, st.session_state.messages, max_tokens)
            except Exception as e:
                answer = f"Erro: {e}"
            st.write(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})

