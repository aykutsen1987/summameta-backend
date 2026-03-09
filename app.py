"""
SummaMeta — Render Backend (Flask)
512 MB RAM ile çalışacak şekilde optimize edilmiştir.
"""

import os
import io
import tempfile
import logging
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
import requests

# ── Opsiyonel PDF / DOCX parse kütüphaneleri ──────────────────────────────────
try:
    import pdfplumber
    PDF_OK = True
except ImportError:
    PDF_OK = False

try:
    from docx import Document as DocxDoc
    DOCX_OK = True
except ImportError:
    DOCX_OK = False

# ── Uygulama kurulumu ──────────────────────────────────────────────────────────
app = Flask(__name__)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── Ortam değişkenleri ─────────────────────────────────────────────────────────
RENDER_SECRET     = os.environ.get("RENDER_SECRET", "")          # Android ile eşleşmeli
GROQ_API_KEY      = os.environ.get("GROQ_API_KEY", "")           # Varsayılan Groq key
TOGETHER_API_KEY  = os.environ.get("TOGETHER_API_KEY", "")
OPENROUTER_API_KEY= os.environ.get("OPENROUTER_API_KEY", "")
CEREBRAS_API_KEY  = os.environ.get("CEREBRAS_API_KEY", "")
GEMINI_API_KEY    = os.environ.get("GEMINI_API_KEY", "")
MAX_FILE_MB       = int(os.environ.get("MAX_FILE_MB", "20"))
MAX_TOKENS        = int(os.environ.get("MAX_TOKENS", "2000"))
MAX_TEXT_CHARS    = 12000   # 512 MB sınırı için metin kırpma

# ── Provider yapılandırmaları ──────────────────────────────────────────────────
PROVIDERS = {
    "groq": {
        "url":    "https://api.groq.com/openai/v1/chat/completions",
        "models": {
            "llama4-scout":  "meta-llama/llama-4-scout-17b-16e-instruct",
            "llama3-70b":    "llama-3.3-70b-versatile",
            "llama3-8b":     "llama-3.1-8b-instant",
            "mixtral-8x7b":  "mixtral-8x7b-32768",
            "gemma3-9b":     "gemma2-9b-it",
        },
        "default_model": "llama4-scout",
        "env_key": GROQ_API_KEY,
    },
    "together": {
        "url":    "https://api.together.xyz/v1/chat/completions",
        "models": {
            "llama3-70b":   "meta-llama/Llama-3.3-70B-Instruct-Turbo",
            "llama3-8b":    "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
            "qwen2-5":      "Qwen/Qwen2.5-72B-Instruct-Turbo",
            "mixtral-8x7b": "mistralai/Mixtral-8x7B-Instruct-v0.1",
            "gemma3-27b":   "google/gemma-3-27b-it",
        },
        "default_model": "llama3-70b",
        "env_key": TOGETHER_API_KEY,
    },
    "openrouter": {
        "url":    "https://openrouter.ai/api/v1/chat/completions",
        "models": {
            "mistral-7b-free":  "mistralai/mistral-7b-instruct:free",
            "llama3-8b-free":   "meta-llama/llama-3-8b-instruct:free",
            "qwen3-8b-free":    "qwen/qwen3-8b:free",
            "deepseek-r1-free": "deepseek/deepseek-r1:free",
            "gemma3-27b-free":  "google/gemma-3-27b-it:free",
        },
        "default_model": "deepseek-r1-free",
        "env_key": OPENROUTER_API_KEY,
    },
    "cerebras": {
        "url":    "https://api.cerebras.ai/v1/chat/completions",
        "models": {
            "llama3-70b-cb":  "llama3.3-70b",
            "llama3-8b-cb":   "llama3.1-8b",
            "deepseek-r1-cb": "deepseek-r1-distill-llama-70b",
            "qwen3-32b-cb":   "qwen-3-32b",
        },
        "default_model": "llama3-70b-cb",
        "env_key": CEREBRAS_API_KEY,
    },
    "gemini": {
        "url":    None,   # Gemini ayrı API kullanıyor, aşağıda özel fonksiyon var
        "models": {
            "gemini-2-flash":   "gemini-2.0-flash",
            "gemini-1-5-flash": "gemini-1.5-flash",
            "gemini-1-5-pro":   "gemini-1.5-pro",
        },
        "default_model": "gemini-2-flash",
        "env_key": GEMINI_API_KEY,
    },
}

# ── Yardımcı fonksiyonlar ──────────────────────────────────────────────────────

def check_render_secret():
    """Android'den gelen X-Render-Secret header'ını doğrular."""
    if not RENDER_SECRET:
        return True  # Ortam değişkeni yoksa geç
    return request.headers.get("X-Render-Secret") == RENDER_SECRET


def extract_text_from_file(file_bytes: bytes, filename: str) -> str:
    """PDF / DOCX / TXT dosyasından metin çıkarır."""
    name = filename.lower()

    if name.endswith(".pdf") and PDF_OK:
        with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
            pages = [p.extract_text() or "" for p in pdf.pages[:30]]  # max 30 sayfa
        return "\n".join(pages)

    if (name.endswith(".docx") or name.endswith(".doc")) and DOCX_OK:
        doc = DocxDoc(io.BytesIO(file_bytes))
        return "\n".join(p.text for p in doc.paragraphs if p.text.strip())

    # TXT veya fallback
    for enc in ("utf-8", "latin-1", "cp1252"):
        try:
            return file_bytes.decode(enc)
        except UnicodeDecodeError:
            continue
    return file_bytes.decode("utf-8", errors="replace")


def build_prompt(text: str, language: str) -> str:
    """Özet istemi oluşturur. 512 MB RAM için metin kırpılır."""
    clipped = text[:MAX_TEXT_CHARS]
    if language == "en":
        return (
            f"Please provide a comprehensive summary of the following document in English. "
            f"Organize it with clear sections, key points, and important details.\n\n"
            f"Document:\n{clipped}"
        )
    return (
        f"Lütfen aşağıdaki belgeyi Türkçe olarak kapsamlı şekilde özetleyin. "
        f"Önemli bölümleri, ana fikirleri ve kritik detayları düzenli şekilde yazın.\n\n"
        f"Belge:\n{clipped}"
    )


def call_openai_compatible(provider_id: str, model_key: str, prompt: str, user_api_key: str) -> str:
    """Groq / Together / OpenRouter / Cerebras için ortak OpenAI-uyumlu çağrı."""
    cfg    = PROVIDERS[provider_id]
    api_key= user_api_key.strip() or cfg["env_key"]
    if not api_key:
        raise ValueError(f"{provider_id} API anahtarı bulunamadı. Ayarlar'dan ekleyin.")

    model_id = cfg["models"].get(model_key, cfg["models"][cfg["default_model"]])
    headers  = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type":  "application/json",
    }
    if provider_id == "openrouter":
        headers["HTTP-Referer"] = "https://summameta.app"
        headers["X-Title"]      = "SummaMeta"

    payload = {
        "model":      model_id,
        "messages":   [{"role": "user", "content": prompt}],
        "max_tokens": MAX_TOKENS,
        "temperature": 0.3,
    }
    resp = requests.post(cfg["url"], json=payload, headers=headers, timeout=90)
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]


def call_gemini(model_key: str, prompt: str, user_api_key: str) -> str:
    """Google Gemini REST API çağrısı."""
    api_key  = user_api_key.strip() or GEMINI_API_KEY
    if not api_key:
        raise ValueError("Gemini API anahtarı bulunamadı.")
    model_id = PROVIDERS["gemini"]["models"].get(model_key, "gemini-2.0-flash")
    url      = (
        f"https://generativelanguage.googleapis.com/v1beta/models/"
        f"{model_id}:generateContent?key={api_key}"
    )
    payload  = {"contents": [{"parts": [{"text": prompt}]}]}
    resp     = requests.post(url, json=payload, timeout=90)
    resp.raise_for_status()
    return resp.json()["candidates"][0]["content"]["parts"][0]["text"]


def call_ollama(model_key: str, prompt: str, base_url: str) -> str:
    """Ollama lokal sunucu çağrısı (base_url kullanıcıdan gelir)."""
    url     = base_url.rstrip("/") + "/api/generate"
    model_map = {
        "llama3-8b":        "llama3.2",
        "mistral-7b":       "mistral",
        "gemma3-4b":        "gemma3:4b",
        "qwen2-5":          "qwen2.5",
        "deepseek-r1-1-5b": "deepseek-r1:1.5b",
    }
    payload = {
        "model":  model_map.get(model_key, "llama3.2"),
        "prompt": prompt,
        "stream": False,
    }
    resp = requests.post(url, json=payload, timeout=120)
    resp.raise_for_status()
    return resp.json().get("response", "")


def summarize(text: str, provider: str, model_key: str, language: str, user_api_key: str) -> str:
    """Seçili sağlayıcı ile özet üretir."""
    prompt = build_prompt(text, language)
    if provider == "gemini":
        return call_gemini(model_key, prompt, user_api_key)
    if provider == "ollama":
        base_url = user_api_key or "http://localhost:11434"
        return call_ollama(model_key, prompt, base_url)
    if provider in PROVIDERS:
        return call_openai_compatible(provider, model_key, prompt, user_api_key)
    raise ValueError(f"Bilinmeyen sağlayıcı: {provider}")


# ── Rotalar ────────────────────────────────────────────────────────────────────

@app.before_request
def auth():
    if request.path in ("/health", "/"):
        return   # Health check'e şifre gerekmez
    if not check_render_secret():
        return jsonify({"error": "Unauthorized"}), 401


@app.route("/health")
def health():
    return jsonify({"status": "ok", "service": "SummaMeta"})


@app.route("/")
def index():
    return jsonify({"service": "SummaMeta API", "version": "1.0.0"})


@app.route("/providers")
def list_providers():
    return jsonify({
        pid: list(cfg["models"].keys())
        for pid, cfg in PROVIDERS.items()
    })


@app.route("/summarize", methods=["POST"])
def summarize_document():
    """Dosya yükleme ile özet üretme."""
    if "file" not in request.files:
        return jsonify({"error": "Dosya bulunamadı"}), 400

    file        = request.files["file"]
    language    = request.form.get("language",    "tr")
    provider    = request.form.get("provider",    "groq")
    model_key   = request.form.get("model_key",   "llama4-scout")
    user_api_key= request.form.get("user_api_key","")

    if file.filename == "":
        return jsonify({"error": "Dosya adı boş"}), 400

    file_bytes = file.read()
    if len(file_bytes) > MAX_FILE_MB * 1024 * 1024:
        return jsonify({"error": f"Dosya {MAX_FILE_MB} MB'dan büyük olamaz"}), 413

    try:
        text = extract_text_from_file(file_bytes, secure_filename(file.filename))
        if not text.strip():
            return jsonify({"error": "Dosyadan metin çıkarılamadı"}), 422

        summary    = summarize(text, provider, model_key, language, user_api_key)
        word_count = len(text.split())
        return jsonify({
            "summary":    summary,
            "word_count": word_count,
            "provider":   provider,
            "model":      model_key,
        })
    except requests.HTTPError as e:
        logger.error(f"API HTTP hatası: {e}")
        return jsonify({"error": f"AI API hatası: {e.response.status_code}"}), 502
    except Exception as e:
        logger.error(f"Özet hatası: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/summarize-text", methods=["POST"])
def summarize_text():
    """Düz metin ile özet üretme."""
    data        = request.form
    text        = data.get("text", "")
    language    = data.get("language",    "tr")
    provider    = data.get("provider",    "groq")
    model_key   = data.get("model_key",   "llama4-scout")
    user_api_key= data.get("user_api_key","")

    if not text.strip():
        return jsonify({"error": "Metin boş"}), 400

    try:
        summary    = summarize(text, provider, model_key, language, user_api_key)
        word_count = len(text.split())
        return jsonify({
            "summary":    summary,
            "word_count": word_count,
            "provider":   provider,
            "model":      model_key,
        })
    except requests.HTTPError as e:
        return jsonify({"error": f"AI API hatası: {e.response.status_code}"}), 502
    except Exception as e:
        logger.error(f"Özet hatası: {e}")
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
