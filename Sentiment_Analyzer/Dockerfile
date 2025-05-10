# immagine base di Python
FROM python:3.9-slim

#  working directory
WORKDIR /app

# Copia il file requirements.txt
COPY app/requirements.txt .

# Installa le dipendenze
RUN pip install --no-cache-dir -r requirements.txt

# Copia il resto del codice
COPY app /app

# Esponi la porta che FastAPI usa
EXPOSE 8000

# Comando per avviare il server
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
