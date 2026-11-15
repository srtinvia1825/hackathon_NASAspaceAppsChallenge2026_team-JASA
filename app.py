import os
import streamlit as st
from dotenv import load_dotenv
from rag_engine import RAGEngine

# Cargar variables de entorno desde .env si existe
load_dotenv()

st.set_page_config(
    page_title="RAG AI - NASA Hackathon 2026",
    page_icon="🚀",
    layout="wide"
)

# Encabezado principal
st.title("🚀 Asistente RAG - NASA Hackathon 2026")
st.caption("Consulta inteligente basada en tu repositorio de datos con Gemini 2.5")

# Barra lateral para configuración
with st.sidebar:
    st.header("⚙️ Configuración")
    
    # 1. API Key de Gemini (detecta .env local o Secrets de Streamlit Cloud)
    env_key = os.getenv("GEMINI_API_KEY", "")
    if not env_key:
        try:
            if "GEMINI_API_KEY" in st.secrets:
                env_key = st.secrets["GEMINI_API_KEY"]
        except Exception:
            pass

    api_key = st.text_input(
        "Gemini API Key",
        value=env_key,
        type="password",
        help="Obtén tu API key gratis en https://aistudio.google.com/"
    )

    if not api_key:
        st.warning("⚠️ Ingresa tu API Key de Gemini para activar el asistente.")
        st.stop()

    st.success("API Key detectada ✅")
    st.divider()

    # 2. Gestión de documentos en data/
    st.subheader("📁 Datos de conocimiento (`data/`)")
    data_dir = "data"
    os.makedirs(data_dir, exist_ok=True)
    files = [f for f in os.listdir(data_dir) if not f.startswith(".")]
    
    if files:
        st.write(f"Archivos encontrados ({len(files)}):")
        for f in files:
            st.markdown(f"- 📄 `{f}`")
    else:
        st.info("Coloca tus archivos (.txt, .md, .pdf, .csv, .json) en la carpeta `data/`.")

    # Botón para reindexar
    reindex = st.button("🔄 Reindexar Documentos", use_container_width=True)

# Inicializar motor RAG en el estado de la sesión
if "rag" not in st.session_state or reindex:
    with st.spinner("Construyendo índice vectorial con Gemini..."):
        try:
            rag = RAGEngine(api_key=api_key)
            num_chunks = rag.load_data(data_dir=data_dir)
            rag.build_index(force_refresh=reindex)
            st.session_state["rag"] = rag
            st.sidebar.success(f"¡Índice listo! {len(rag.chunks)} fragmentos cargados.")
        except Exception as e:
            st.error(f"Error inicializando RAG: {e}")
            st.stop()

# Historial de conversación
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "¡Hola! Soy tu asistente para el Hackathon NASA 2026. Pregúntame cualquier cosa sobre los datos que has cargado."}
    ]

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "sources" in msg and msg["sources"]:
            with st.expander("📚 Fuentes consultadas"):
                for chunk, score in msg["sources"]:
                    st.caption(f"**{chunk['source']}** (Similitud: {score:.2f})")
                    st.text(chunk["text"][:300] + ("..." if len(chunk["text"]) > 300 else ""))

# Input del usuario
if user_prompt := st.chat_input("Escribe tu pregunta sobre los datos..."):
    # Guardar y mostrar mensaje del usuario
    st.session_state.messages.append({"role": "user", "content": user_prompt})
    with st.chat_message("user"):
        st.markdown(user_prompt)

    # Generar respuesta con RAG
    with st.chat_message("assistant"):
        with st.spinner("Buscando en tus documentos y consultando a Gemini..."):
            try:
                rag: RAGEngine = st.session_state["rag"]
                result = rag.query(user_prompt, top_k=4)
                
                st.markdown(result["answer"])
                
                if result["sources"]:
                    with st.expander("📚 Fuentes consultadas"):
                        for chunk, score in result["sources"]:
                            st.caption(f"**{chunk['source']}** (Similitud: {score:.2f})")
                            st.text(chunk["text"][:300] + ("..." if len(chunk["text"]) > 300 else ""))

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": result["answer"],
                    "sources": result["sources"]
                })
            except Exception as e:
                error_msg = f"Ocurrió un error al procesar tu consulta: {e}"
                st.error(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})
