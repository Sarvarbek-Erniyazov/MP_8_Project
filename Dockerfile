FROM python:3.11-slim

WORKDIR /app

# LightGBM uchun tizim paketlari
RUN apt-get update && apt-get install -y \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Hamma fayllarni konteynerning /app papkasiga nusxalaydi
COPY . .

EXPOSE 8000

# DIQQAT: Agar main.py fayli app/ papkasi ichida bo'lsa:
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]