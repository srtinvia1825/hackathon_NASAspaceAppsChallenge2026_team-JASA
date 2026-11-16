import os
import streamlit as st
from dotenv import load_dotenv
from rag_engine import RAGEngine

# Cargar variables de entorno desde .env si existe
load_dotenv()

st.set_page_config(
    page_title="Team JASA - Hackathon NASA space apps challenge 2026 - Flame in free fall challenge",
    page_icon="🚀",
    layout="wide"
)

# Encabezado principal
st.title("AI assistant - Flame in free fall challenge")
st.caption("AI organizer for the NASA PSI documents")

# Barra lateral para configuración
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # 1. API Key de Gemini (detecta .env local o Secrets de Streamlit Cloud)
    env_key = os.getenv("GEMINI_API_KEY", "")
    if not env_key:
        try:
            if "GEMINI_API_KEY" in st.secrets:
                env_key = st.secrets["GEMINI_API_KEY"]
        except Exception:
            pass

    api_key = env_key

    if not api_key:
        st.warning("⚠️ Enter your Gemini API Key to activate the assistant.")
        st.stop()

    st.divider()

    # 2. Explorador de documentos en data/
    data_dir = "data"
    os.makedirs(data_dir, exist_ok=True)
    category_names = [
        "Archived Combustion Science",
        "Multiphysics CFD Analysis of Thermo-Convective Effects on Cool Flame Dynamics",
    ]
    categories = {}
    for folder in category_names:
        category_dir = os.path.join(data_dir, folder)
        categories[folder] = sorted(
            os.path.relpath(os.path.join(root, filename), category_dir)
            for root, _, filenames in os.walk(category_dir)
            for filename in filenames
            if filename.lower().endswith(".pdf") and not filename.startswith(".")
        )

    logo_col, title_col = st.columns([1, 4])

    with logo_col:
        st.image("assets/nasa-logo.png", width=36)

    with title_col:
        st.markdown(
            "<h3 style='text-align: left;'>PSI data</h3>",
            unsafe_allow_html=True
        )

    for folder, folder_files in categories.items():
        with st.expander(f"📁 {folder}", expanded=False):
            if folder_files:
                for filename in folder_files:
                    st.markdown(f"- 📄 `{filename}`")
            else:
                st.info("No PDF files found in this folder.")
    
    # Botón para reindexar
    reindex = st.button("Reindex Documents", use_container_width=True)

# Inicializar motor RAG en el estado de la sesión
if "rag" not in st.session_state or reindex:
    with st.spinner("Loading documents and building index..."):
        try:
            rag = RAGEngine(api_key=api_key)
            rag.load_data(data_dir=data_dir)
            rag.build_index(force_refresh=reindex)
            st.session_state["rag"] = rag
        except Exception as e:
            st.error(f"Error critical initializing RAG: {e}")
            st.stop()

# Historial de conversación
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello! I'm your AI assistant for the NASA PSI documents. Ask me anything about the loaded data"}
    ]

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "sources" in msg and msg["sources"]:
            with st.expander("📚 Checked sources"):
                for chunk, score in msg["sources"]:
                    st.caption(f"**{chunk['source']}** (Similitud: {score:.2f})")
                    st.text(chunk["text"][:300] + ("..." if len(chunk["text"]) > 300 else ""))

# Input del usuario
if user_prompt := st.chat_input("Ask anything about the loaded data..."):
    # Guardar y mostrar mensaje del usuario
    st.session_state.messages.append({"role": "user", "content": user_prompt})
    with st.chat_message("user"):
        st.markdown(user_prompt)

    # Generar respuesta con RAG
    with st.chat_message("assistant"):
        with st.spinner("Searching data..."):
            try:
                rag: RAGEngine = st.session_state["rag"]
                result = rag.query(user_prompt, top_k=4)
                
                st.markdown(result["answer"])
                
                if result["sources"]:
                    with st.expander("📚 Checked sources"):
                        for chunk, score in result["sources"]:
                            st.caption(f"**{chunk['source']}** (Similitud: {score:.2f})")
                            st.text(chunk["text"][:300] + ("..." if len(chunk["text"]) > 300 else ""))

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": result["answer"],
                    "sources": result["sources"]
                })
            except Exception as e:
                error_msg = f"An error occurred while processing your query: {e}"
                st.error(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})
                
