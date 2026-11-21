# Mech Royale bot (mechfight/1). Build: docker build -t mech-bot .
# Runs with no network, read-only root (writable /tmp), 1 CPU, 512 MB, map at /arena, uid 65534.
# Standard library only: nothing to install. The map is also bundled (bot/arena) as a fallback.
FROM python:3.13.18-slim@sha256:e48f59cdba69a3cbd9ed2cf0f348d90d45a8d350d98134cf1070bc106ee9ed90
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /bot
COPY bot/ /bot/
RUN python3 -m compileall -q /bot && chmod -R a+rX /bot
ENTRYPOINT ["python3", "-u", "/bot/main.py"]
