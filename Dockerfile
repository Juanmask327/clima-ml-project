# ============================================================
# PREDICCIÓN CLIMÁTICA ML · Imagen Docker
# ============================================================
FROM python:3.12-slim

ENV PYTHONUTF8=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Dependencias primero (capa cacheable)
COPY Requirements.txt .
RUN pip install --no-cache-dir -r Requirements.txt

# Código
COPY backend/ ./backend/
COPY frontend/ ./frontend/

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=5).status==200 else 1)"

CMD ["uvicorn", "api.main:app", "--app-dir", "backend", "--host", "0.0.0.0", "--port", "8000"]
