# CryptoMind — production container (paper/shadow mode).
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# deps first for layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# app: the code is kept in /src and copied over /app on EVERY start. /app is
# the volume (state.json, db, reports/, .cache/ live there, next to the code),
# so a rebuilt image's code replaces the old copy while state persists. The
# state files are never in /src (.dockerignore).
COPY . /src

# non-root user
RUN useradd -m -u 10001 cryptomind && mkdir -p /app && chown -R cryptomind:cryptomind /app /src
USER cryptomind
VOLUME ["/app"]

EXPOSE 8000

# Bind to localhost by default INSIDE the container is wrong (nothing could
# reach it); we bind 0.0.0.0 and rely on the reverse proxy / compose network
# + API token for protection. Set CRYPTOMIND_API_TOKEN in the environment.
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/api/status',timeout=4).status==200 else 1)"

CMD ["sh", "-c", "cp -a /src/. /app/ && exec uvicorn app.main:app --host 0.0.0.0 --port 8000"]
