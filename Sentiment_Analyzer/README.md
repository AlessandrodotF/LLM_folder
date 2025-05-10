# Sentiment Analysis API

Questo progetto fornisce una semplice API REST per l'analisi del sentiment di testi, sviluppata con FastAPI e Pydantic.

## Caratteristiche

- Endpoint di *health check* per verificare che l'API sia attiva
- Endpoint `/predict` per ottenere il sentiment (etichetta e punteggio) di un testo
- Architettura modulare: la logica di previsione è separata in `model.py`

## Struttura del progetto

```text
.
├── app.py            # Definizione dell'applicazione FastAPI
├── model.py          # Funzione predict_sentiment(text) che restituisce {'label', 'score'}
├── requirements.txt  # Dipendenze Python
├── README.md         # Questo file
└── .env              # Variabili d'ambiente (opzionale)
```

## Variabili d'ambiente

- `PORT` – Porta sulla quale esporre l'API (default: `8000`)

## Utilizzo

Avvia l'API in locale:

```bash
uvicorn app:app --reload --host 0.0.0.0 --port ${PORT:-8000}
```

### Endpoint

1. **Health check**
   ```http
   GET /
   ```
   **Response**:
   ```json
   { "message": "API is live!" }
   ```

2. **Previsione sentiment**
   ```http
   POST /predict
   Content-Type: application/json

   { "text": "Il tuo testo qui" }
   ```
   **Response**:
   ```json
   {
     "label": "POSITIVE",    # o "NEGATIVE", etc.
     "score": 0.95           # probabilità o confidenza
   }
   ```

## Docker (opzionale)

1. Costruisci l'immagine Docker:
   ```bash
   docker build -t sentiment-api .
   ```
2. Esegui il container:
   ```bash
   docker run -d -p 8000:8000 --env PORT=8000 sentiment-api
   ```

L'API sarà disponibile su `http://localhost:8000`.

