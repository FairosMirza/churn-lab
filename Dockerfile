# Churn Lab — deployable on Hugging Face Spaces (Docker), Render, Railway, Fly.io
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY generate_data.py experiments.py narrator.py server.py ./
COPY web/ web/

# HF Spaces expects 7860; Render/Railway inject $PORT
EXPOSE 7860
CMD uvicorn server:app --host 0.0.0.0 --port ${PORT:-7860}
