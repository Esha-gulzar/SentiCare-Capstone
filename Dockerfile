# ── Stage 1: Build React frontend ─────────────────────────────────────────────
FROM node:18-slim AS frontend-build

WORKDIR /frontend
COPY senticare-frontend/package*.json ./
RUN npm install
COPY senticare-frontend/ ./
RUN npm run build

# ── Stage 2: Python backend + serve React static files ────────────────────────
FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y ffmpeg && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all backend code
COPY . .

# Copy the React build output into Flask's static folder
COPY --from=frontend-build /frontend/build ./static

EXPOSE 7860

CMD ["gunicorn", "--bind", "0.0.0.0:7860", "--timeout", "120", "app:app"]