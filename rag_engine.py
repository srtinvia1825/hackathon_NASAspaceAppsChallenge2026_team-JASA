import os
import glob
import pickle
import numpy as np
from typing import List, Dict, Tuple
from google import genai
from google.genai import types

try:
    from pypdf import PdfReader
except ImportError:
    PdfReader = None

CACHE_FILE = "index_cache/embeddings.pkl"

def extract_text_from_file(file_path: str) -> str:
    """Extrae texto de archivos .txt, .md, .pdf, .csv o .json."""
    ext = os.path.splitext(file_path)[1].lower()
    try:
        if ext in [".txt", ".md", ".csv", ".json"]:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                return f.read()
        elif ext == ".pdf":
            if PdfReader is None:
                return ""
            reader = PdfReader(file_path)
            text = []
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text.append(page_text)
            return "\n".join(text)
    except Exception as e:
        print(f"Error leyendo {file_path}: {e}")
    return ""

def chunk_text(text: str, source: str, chunk_size: int = 800, overlap: int = 150) -> List[Dict]:
    """Divide el texto en fragmentos con solapamiento para no perder contexto."""
    chunks = []
    text = text.strip()
    if not text:
        return chunks
    
    start = 0
    idx = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append({
            "id": f"{source}_{idx}",
            "source": source,
            "text": chunk.strip()
        })
        idx += 1
        start += chunk_size - overlap
        if end >= len(text):
            break
    return chunks

class RAGEngine:
    def __init__(self, api_key: str):
        self.api_key = api_key
        # Usamos v1beta donde text-embedding-004 está siempre disponible
        try:
            self.client = genai.Client(
                api_key=api_key,
                http_options=types.HttpOptions(api_version="v1beta")
            )
        except Exception:
            self.client = genai.Client(api_key=api_key)
        self.chunks: List[Dict] = []
        self.embeddings: np.ndarray = np.array([])
        self.embedding_model = "text-embedding-004"

    def _embed_batch(self, batch):
        """Genera embeddings con fallback automático si un modelo no está disponible."""
        try:
            return self.client.models.embed_content(
                model=self.embedding_model,
                contents=batch,
            )
        except Exception as e:
            err_str = str(e).lower()
            if "not found" in err_str and self.embedding_model != "embedding-001":
                print(f"[Aviso] {self.embedding_model} no encontrado, usando embedding-001 como respaldo.")
                self.embedding_model = "embedding-001"
                return self.client.models.embed_content(
                    model="embedding-001",
                    contents=batch,
                )
            raise e

    def load_data(self, data_dir: str = "data") -> int:
        """Carga documentos de la carpeta data/ y los fragmenta."""
        all_chunks = []
        if not os.path.exists(data_dir):
            os.makedirs(data_dir, exist_ok=True)
            return 0

        # Buscar todos los archivos en data/ y subcarpetas
        files = glob.glob(os.path.join(data_dir, "**", "*.*"), recursive=True)
        valid_exts = {".txt", ".md", ".pdf", ".csv", ".json"}
        valid_files = [f for f in files if os.path.splitext(f)[1].lower() in valid_exts]

        for file_path in valid_files:
            rel_name = os.path.relpath(file_path, data_dir)
            content = extract_text_from_file(file_path)
            if content.strip():
                file_chunks = chunk_text(content, source=rel_name)
                all_chunks.extend(file_chunks)

        self.chunks = all_chunks
        return len(self.chunks)

    def build_index(self, force_refresh: bool = False):
        """Genera o carga los embeddings de los fragmentos usando text-embedding-004."""
        os.makedirs("index_cache", exist_ok=True)

        if not force_refresh and os.path.exists(CACHE_FILE):
            try:
                with open(CACHE_FILE, "rb") as f:
                    data = pickle.load(f)
                    self.chunks = data["chunks"]
                    self.embeddings = data["embeddings"]
                    print(f"Caché cargado con {len(self.chunks)} fragmentos.")
                    return
            except Exception as e:
                print(f"Error cargando caché, recalculando: {e}")

        if not self.chunks:
            self.load_data()

        if not self.chunks:
            self.embeddings = np.array([])
            return

        texts = [c["text"] for c in self.chunks]
        
        # Procesar en lotes de hasta 50 para respetar límites de la API
        all_vectors = []
        batch_size = 50
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            response = self._embed_batch(batch)
            for emb in response.embeddings:
                all_vectors.append(emb.values)

        self.embeddings = np.array(all_vectors)

        with open(CACHE_FILE, "wb") as f:
            pickle.dump({"chunks": self.chunks, "embeddings": self.embeddings}, f)
        print("Nuevo índice vectorial guardado con éxito.")

    def search(self, query: str, top_k: int = 4) -> List[Tuple[Dict, float]]:
        """Busca los fragmentos más relevantes usando similitud coseno."""
        if len(self.embeddings) == 0 or not self.chunks:
            return []

        # Vectorizar la pregunta con fallback
        resp = self._embed_batch([query])
        query_vec = np.array(resp.embeddings[0].values)

        # Similitud coseno matricial con NumPy
        dot_product = np.dot(self.embeddings, query_vec)
        norms = np.linalg.norm(self.embeddings, axis=1) * np.linalg.norm(query_vec)
        scores = dot_product / (norms + 1e-10)

        top_indices = np.argsort(scores)[::-1][:top_k]
        results = [(self.chunks[i], float(scores[i])) for i in top_indices]
        return results

    def query(self, question: str, top_k: int = 4) -> Dict:
        """Flujo completo de RAG: búsqueda de contexto + respuesta con Gemini."""
        retrieved = self.search(question, top_k=top_k)
        
        if not retrieved:
            context_str = "No hay información disponible en la base de datos local."
        else:
            context_pieces = [
                f"[Fuente: {chunk['source']}]\n{chunk['text']}"
                for chunk, score in retrieved
            ]
            context_str = "\n\n---\n\n".join(context_pieces)

        system_instruction = (
            "Eres un asistente de inteligencia artificial especializado para el Hackathon NASA 2026. "
            "Tu misión es responder preguntas del usuario fundamentándote exclusivamente en el contexto proporcionado. "
            "Si la información necesaria para responder no está en el contexto, indícalo con cortesía "
            "y aclara qué es lo que sí se sabe según los datos disponibles. "
            "Sé preciso, conciso y profesional. Cita las fuentes cuando sea relevante."
        )

        prompt = f"""CONTEXTO RECUPERADO DE LOS DATOS:
{context_str}

PREGUNTA DEL USUARIO:
{question}
"""

        response = None
        models_to_try = ["gemini-2.5-flash", "gemini-2.0-flash", "gemini-1.5-flash"]
        last_err = None
        for model_name in models_to_try:
            try:
                response = self.client.models.generate_content(
                    model=model_name,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        system_instruction=system_instruction,
                        temperature=0.2,
                    )
                )
                break
            except Exception as e:
                last_err = e
                if "not found" in str(e).lower():
                    continue
                raise e

        if response is None:
            raise last_err or Exception("No se pudo conectar con ningún modelo de Gemini.")

        return {
            "answer": response.text,
            "sources": retrieved
        }
