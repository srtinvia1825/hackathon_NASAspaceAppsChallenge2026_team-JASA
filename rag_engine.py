import os
import re
import glob
import math
import pickle
import numpy as np
from collections import Counter
from typing import List, Dict, Tuple
from google import genai
from google.genai import types
from docx import Document
from openpyxl import load_workbook
from pptx import Presentation

try:
    from pypdf import PdfReader
except ImportError:
    PdfReader = None

CACHE_FILE = "index_cache/embeddings.pkl"

def tokenize(text: str) -> List[str]:
    """Tokeniza texto separando palabras limpias."""
    return re.findall(r'\b[a-zA-Z0-9áéíóúÁÉÍÓÚñÑüÜ]{2,}\b', text.lower())

def extract_text_from_file(file_path: str) -> str:
    """Extrae texto de archivos de texto, PDF, CSV y documentos de Office."""
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
        elif ext == ".xlsx":
            workbook = load_workbook(file_path, read_only=True, data_only=True)
            try:
                text = []
                for worksheet in workbook.worksheets:
                    text.append(f"[Sheet: {worksheet.title}]")
                    for row in worksheet.iter_rows(values_only=True):
                        values = [str(value) for value in row if value is not None]
                        if values:
                            text.append("\t".join(values))
                return "\n".join(text)
            finally:
                workbook.close()
        elif ext == ".docx":
            document = Document(file_path)
            text = [paragraph.text for paragraph in document.paragraphs if paragraph.text.strip()]
            for table in document.tables:
                for row in table.rows:
                    text.append("\t".join(cell.text for cell in row.cells))
            return "\n".join(text)
        elif ext == ".pptx":
            presentation = Presentation(file_path)
            text = []
            for slide_number, slide in enumerate(presentation.slides, start=1):
                text.append(f"[Slide {slide_number}]")
                for shape in slide.shapes:
                    if shape.has_text_frame and shape.text.strip():
                        text.append(shape.text)
                    if shape.has_table:
                        for row in shape.table.rows:
                            text.append("\t".join(cell.text for cell in row.cells))
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

class LocalBM25Index:
    """Motor de búsqueda BM25 local (estándar de la industria, 0 dependencias externas, nunca falla con 404)."""
    def __init__(self, chunks: List[Dict]):
        self.chunks = chunks
        self.doc_tokens = [tokenize(c["text"]) for c in chunks]
        self.doc_lens = [len(tokens) for tokens in self.doc_tokens]
        self.avg_doc_len = sum(self.doc_lens) / max(len(self.doc_lens), 1)
        self.doc_counts = Counter()
        for tokens in self.doc_tokens:
            for t in set(tokens):
                self.doc_counts[t] += 1
        self.n_docs = len(chunks)

    def search(self, query: str, top_k: int = 4) -> List[Tuple[Dict, float]]:
        query_tokens = tokenize(query)
        if not query_tokens or self.n_docs == 0:
            return [(self.chunks[i], 1.0) for i in range(min(top_k, len(self.chunks)))]

        scores = []
        k1 = 1.5
        b = 0.75
        for i, tokens in enumerate(self.doc_tokens):
            term_freq = Counter(tokens)
            score = 0.0
            doc_len = self.doc_lens[i]
            for q in query_tokens:
                if q in term_freq:
                    df = self.doc_counts.get(q, 0)
                    idf = math.log((self.n_docs - df + 0.5) / (df + 0.5) + 1.0)
                    tf = term_freq[q]
                    numerator = tf * (k1 + 1)
                    denominator = tf + k1 * (1 - b + b * (doc_len / max(self.avg_doc_len, 1)))
                    score += idf * (numerator / denominator)
            scores.append((self.chunks[i], score))

        scores.sort(key=lambda x: x[1], reverse=True)
        # Normalizar scores para vista de usuario
        max_score = scores[0][1] if scores and scores[0][1] > 0 else 1.0
        return [(chunk, round(score / max_score, 2)) for chunk, score in scores[:top_k]]

class RAGEngine:
    def __init__(self, api_key: str):
        self.api_key = api_key
        try:
            self.client = genai.Client(
                api_key=api_key,
                http_options=types.HttpOptions(api_version="v1beta")
            )
        except Exception:
            self.client = genai.Client(api_key=api_key)
        self.chunks: List[Dict] = []
        self.embeddings: np.ndarray = np.array([])
        self.bm25_index: LocalBM25Index = None
        self.use_local_bm25 = False
        self.data_loaded = False

    def load_data(self, data_dir: str = "data") -> int:
        """Carga documentos de una o varias carpetas/archivos y los fragmenta."""
        all_chunks = []
        if isinstance(data_dir, (str, os.PathLike)):
            data_sources = [os.fspath(data_dir)]
        else:
            data_sources = [os.fspath(source) for source in data_dir]

        if len(data_sources) == 1 and not os.path.exists(data_sources[0]):
            os.makedirs(data_sources[0], exist_ok=True)

        valid_exts = {".txt", ".md", ".pdf", ".csv", ".json", ".xlsx", ".docx", ".pptx"}
        valid_files = []
        seen_files = set()
        include_source_dir = len(data_sources) > 1

        for source in data_sources:
            if os.path.isfile(source):
                source_dir = os.path.dirname(source)
                files = [source]
            elif os.path.isdir(source):
                source_dir = source
                files = glob.glob(os.path.join(source, "**", "*.*"), recursive=True)
            else:
                continue

            for file_path in files:
                if os.path.splitext(file_path)[1].lower() not in valid_exts:
                    continue
                file_key = os.path.normcase(os.path.abspath(file_path))
                if file_key in seen_files:
                    continue
                seen_files.add(file_key)
                rel_name = os.path.relpath(file_path, source_dir)
                if include_source_dir:
                    source_name = os.path.basename(os.path.normpath(source_dir if os.path.isfile(source) else source))
                    rel_name = os.path.join(source_name, rel_name)
                valid_files.append((file_path, rel_name))

        for file_path, rel_name in valid_files:
            content = extract_text_from_file(file_path)
            if content.strip():
                file_chunks = chunk_text(content, source=rel_name)
                all_chunks.extend(file_chunks)

        self.chunks = all_chunks
        self.bm25_index = LocalBM25Index(self.chunks)
        self.data_loaded = True
        return len(self.chunks)

    def build_index(self, force_refresh: bool = False):
        """Genera el índice de búsqueda con embeddings o BM25. NUNCA lanza excepción."""
        try:
            os.makedirs("index_cache", exist_ok=True)
        except Exception:
            pass

        if not self.data_loaded:
            self.load_data()

        # Siempre construir BM25 de respaldo primero (nunca falla)
        self.bm25_index = LocalBM25Index(self.chunks)

        if not self.chunks:
            return

        # Intentar cargar caché local previo
        if not force_refresh and os.path.exists(CACHE_FILE):
            try:
                with open(CACHE_FILE, "rb") as f:
                    data = pickle.load(f)
                    cached_chunks = data.get("chunks", [])
                    cached_embs = data.get("embeddings", np.array([]))
                    if len(cached_chunks) > 0 and len(cached_embs) > 0:
                        self.chunks = cached_chunks
                        self.embeddings = cached_embs
                        self.bm25_index = LocalBM25Index(self.chunks)
                        return
            except Exception:
                pass

        # Intentar generar embeddings en Gemini (opcional - si falla, BM25 ya está listo)
        texts = [c["text"] for c in self.chunks]
        batch_size = 50
        embedding_models = ["text-embedding-004", "embedding-001"]

        success = False
        for emb_model in embedding_models:
            try:
                all_vectors = []
                for i in range(0, len(texts), batch_size):
                    batch = texts[i:i + batch_size]
                    response = self.client.models.embed_content(
                        model=emb_model,
                        contents=batch,
                    )
                    for emb in response.embeddings:
                        all_vectors.append(emb.values)
                self.embeddings = np.array(all_vectors)
                success = True
                print(f"[RAG] Embeddings generados con éxito usando {emb_model}.")
                break
            except Exception as e:
                print(f"[Aviso] {emb_model} no disponible: {e}")
                continue

        if not success:
            # Fallback automático: activar motor de búsqueda BM25 local
            print("[RAG] Activando motor BM25 local de alta precisión (sin dependencia de API de embeddings).")
            self.use_local_bm25 = True
            self.embeddings = np.array([])

        try:
            with open(CACHE_FILE, "wb") as f:
                pickle.dump({"chunks": self.chunks, "embeddings": self.embeddings}, f)
        except Exception:
            pass

    def search(self, query: str, top_k: int = 4) -> List[Tuple[Dict, float]]:
        """Busca fragmentos relevantes usando vectores o BM25."""
        if not self.chunks:
            return []

        # Si estamos en modo BM25 local o no hay embeddings calculados
        if self.use_local_bm25 or len(self.embeddings) == 0:
            if not self.bm25_index:
                self.bm25_index = LocalBM25Index(self.chunks)
            return self.bm25_index.search(query, top_k=top_k)

        # Si tenemos embeddings de Google
        try:
            resp = self.client.models.embed_content(
                model="text-embedding-004",
                contents=[query]
            )
            query_vec = np.array(resp.embeddings[0].values)
            dot_product = np.dot(self.embeddings, query_vec)
            norms = np.linalg.norm(self.embeddings, axis=1) * np.linalg.norm(query_vec)
            scores = dot_product / (norms + 1e-10)
            top_indices = np.argsort(scores)[::-1][:top_k]
            return [(self.chunks[i], float(scores[i])) for i in top_indices]
        except Exception:
            # Si falla la llamada de embedding en caliente, usar BM25
            if not self.bm25_index:
                self.bm25_index = LocalBM25Index(self.chunks)
            return self.bm25_index.search(query, top_k=top_k)

    def query(self, question: str, top_k: int = 4) -> Dict:
        """Flujo completo de RAG: búsqueda de contexto + respuesta con Gemini."""
        retrieved = self.search(question, top_k=top_k)

        if not retrieved:
            context_str = "No hay información disponible en los documentos locales."
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
        models_to_try = ["gemini-3.8-flash", "gemini-2.0-flash", "gemini-1.5-flash"]
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
