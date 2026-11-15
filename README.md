# 🚀 NASA Hackathon 2026 - Asistente RAG con Gemini y Streamlit

Sistema de Inteligencia Artificial RAG (Retrieval-Augmented Generation) para responder preguntas fundamentadas en conjuntos de datos y documentos locales usando la API de Gemini 2.5.

## 📋 Estructura del Proyecto

```text
├── data/                  <- Carpeta donde colocas tus archivos (.txt, .md, .pdf, .csv, .json)
│   └── ejemplo_nasa.txt   <- Archivo de prueba
├── agent.py               <- Ejecución en consola / terminal
├── app.py                 <- Aplicación web interactiva con Streamlit
├── rag_engine.py          <- Motor RAG (embeddings, búsqueda coseno matricial y Gemini)
├── requirements.txt       <- Dependencias del proyecto
├── .gitignore             <- Protege tu API key y cachés locales
└── .env.example           <- Plantilla de variables de entorno
```

## 🛠️ Instalación y Uso Local

### 1. Clonar o abrir el proyecto
Asegúrate de estar en la carpeta del proyecto en tu terminal.

### 2. Configurar la API Key
Obtén tu clave gratuita en [Google AI Studio](https://aistudio.google.com/).
Crea un archivo llamado `.env` en la raíz del proyecto con el siguiente contenido:

```bash
GEMINI_API_KEY=tu_api_key_aqui
```

### 3. Instalar dependencias
```bash
pip install -r requirements.txt
```

### 4. Ejecutar la Aplicación Web
```bash
streamlit run app.py
```
Se abrirá automáticamente tu navegador en `http://localhost:8501`.

---

## 🌐 Publicación en la Web (Gratis y Pública para todo el mundo)

1. **Subir a GitHub**:
   * Inicializa tu repositorio Git:
     ```bash
     git init
     git add .
     git commit -m "Initial commit - NASA Hackathon RAG"
     ```
   * Súbelo a tu repositorio de GitHub. *(Nota: El archivo `.gitignore` ya está configurado para que tu clave secreta `.env` nunca se suba)*.

2. **Desplegar en Streamlit Community Cloud**:
   * Entra en [share.streamlit.io](https://share.streamlit.io/) e inicia sesión con tu cuenta de GitHub.
   * Haz clic en **"New app"**.
   * Selecciona tu repositorio y la rama `main`, indicando que el archivo principal es `app.py`.
   * En la sección **Advanced settings -> Secrets**, añade tu clave:
     ```toml
     GEMINI_API_KEY = "tu_clave_real_aqui"
     ```
   * Haz clic en **Deploy**. ¡En 1 minuto tendrás un enlace público `https://tu-app.streamlit.app` accesible por cualquier persona!
