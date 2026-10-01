from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, List
from rag_engine import ChatNutriRAG, interroga_database_tabulare

app = FastAPI(title="NutriRAG API")

# Cache per non ricaricare i modelli a ogni messaggio
bot_instances = {}

class ChatRequest(BaseModel):
    user_query: str
    profilo_utente: str = "Nessuna preferenza specifica"
    storico: str = ""
    immagine_b64: Optional[str] = None
    mime_type: str = "image/jpeg"
    groq_key: str
    gemini_key: Optional[str] = None
    is_tabular: bool = False

class ChatResponse(BaseModel):
    risposta: str
    alimenti_estratti: List[str]

@app.post("/api/v1/chat", response_model=ChatResponse)
def chat_endpoint(req: ChatRequest):
    # Routing Tabulare
    if req.is_tabular:
        risposta = interroga_database_tabulare(req.user_query, req.groq_key)
        return {"risposta": risposta, "alimenti_estratti": []}
        
    # Routing RAG
    if req.groq_key not in bot_instances:
        bot_instances[req.groq_key] = ChatNutriRAG(req.groq_key, req.gemini_key)
    bot = bot_instances[req.groq_key]
    
    try:
        risposta = bot.gestisci_chat(
            user_query=req.user_query,
            profilo_utente=req.profilo_utente,
            storico=req.storico,
            immagine_b64=req.immagine_b64,
            mime_type=req.mime_type
        )
        alimenti = getattr(bot, "ultimi_alimenti", [])
        return {"risposta": risposta, "alimenti_estratti": alimenti}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))