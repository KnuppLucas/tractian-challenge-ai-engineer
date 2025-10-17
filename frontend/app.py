import streamlit as st
import requests
import time

BACKEND_URL = "http://web:8000/api"

st.set_page_config(page_title="RAG PDF QA System", page_icon="📚", layout="centered")

st.title("📚 RAG PDF QA System")
st.caption("Query PDFs using a local RAG pipeline with FAISS + HuggingFace")

if "uploaded" not in st.session_state:
    st.session_state.uploaded = False
if "processing" not in st.session_state:
    st.session_state.processing = False

def fake_progress(duration_sec, placeholder, text):
    steps = 100
    for i in range(steps + 1):
        time.sleep(duration_sec / steps)
        placeholder.progress(i)
        placeholder.caption(f"{text} ({i}%)")
    placeholder.empty()

st.header("📤 Upload de PDFs")
uploaded_files = st.file_uploader(
    "Selecione um ou mais arquivos PDF",
    type=["pdf"],
    accept_multiple_files=True
)

upload_disabled = st.session_state.processing
if st.button("📨 Enviar PDFs", disabled=upload_disabled):
    if not uploaded_files:
        st.warning("Por favor, selecione pelo menos um arquivo PDF.")
    else:
        st.session_state.processing = True
        progress_placeholder = st.empty()
        progress_placeholder.info("⏳ Iniciando upload...")

        try:
            files_payload = [
                ("files", (f.name, f.getvalue(), "application/pdf")) for f in uploaded_files
            ]
            start_time = time.time()
            duration_fake = 10
            response = requests.post(f"{BACKEND_URL}/documents/", files=files_payload, timeout=None)

            while time.time() - start_time < duration_fake:
                elapsed = time.time() - start_time
                pct = int((elapsed / duration_fake) * 100)
                progress_placeholder.progress(pct)
                progress_placeholder.caption(f"Processando PDFs ({pct}%)")
                time.sleep(0.1)
            progress_placeholder.empty()

            if response.status_code in [200, 202]:
                data = response.json()
                st.session_state.uploaded = True
                st.success(f"✅ {data.get('documents_indexed', 0)} documentos processados! ({data.get('total_chunks', 0)} chunks)")
            else:
                st.error(f"Erro ao enviar PDFs: {response.status_code}\n{response.text}")

        except Exception as e:
            progress_placeholder.empty()
            st.error(f"Erro durante upload: {e}")

        finally:
            st.session_state.processing = False

st.header("💬 Fazer Pergunta")
question = st.text_area("Digite sua pergunta sobre os documentos:")

ask_disabled = st.session_state.processing
if st.button("🚀 Perguntar", disabled=ask_disabled):
    if not question.strip():
        st.warning("Digite uma pergunta antes de enviar.")
    else:
        st.session_state.processing = True
        progress_placeholder = st.empty()
        progress_placeholder.info("⏳ Consultando RAG...")

        try:
            start_time = time.time()
            duration_fake = 10
            response = requests.post(f"{BACKEND_URL}/question/", json={"question": question}, timeout=None)

            while time.time() - start_time < duration_fake:
                elapsed = time.time() - start_time
                pct = int((elapsed / duration_fake) * 100)
                progress_placeholder.progress(pct)
                progress_placeholder.caption(f"Consultando RAG ({pct}%)")
                time.sleep(0.1)
            progress_placeholder.empty()

            if response.status_code in [200, 202]:
                data = response.json()
                answer = data.get("answer", "Sem resposta.")
                refs = data.get("references", [])
                st.success(answer)
                if refs:
                    with st.expander("📚 Referências"):
                        for r in refs:
                            st.write(f"- {r}")
            else:
                st.error(f"Erro {response.status_code}: {response.text}")

        except Exception as e:
            progress_placeholder.empty()
            st.error(f"Erro durante consulta: {e}")

        finally:
            st.session_state.processing = False
