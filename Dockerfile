FROM python:3.11-slim
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*
COPY . /app/
RUN python -m venv .venv && .venv/bin/pip install --quiet flask numpy 2>/dev/null || true
EXPOSE 8080
CMD [".venv/bin/python", "backend/api/app.py"]
