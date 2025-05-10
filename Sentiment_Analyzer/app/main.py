from fastapi import FastAPI
from pydantic import BaseModel
import os
import uvicorn
from model import predict_sentiment  

# Crea l'istanza dell'app FastAPI
app = FastAPI()

# Aggiungi una route per la root
@app.get("/")
def read_root():
    return {"message": "API is live!"}

# Crea il modello per il corpo della richiesta
class TextRequest(BaseModel):
    text: str

# Definisci l'endpoint per la previsione del sentiment
@app.post("/predict")
def predict(request: TextRequest):
    sentiment = predict_sentiment(request.text)
    return {"label": sentiment['label'], "score": sentiment['score']}

# Avvia il server solo se questo è il file principale
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8000)))  
