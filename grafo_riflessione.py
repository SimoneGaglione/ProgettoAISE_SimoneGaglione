from typing import TypedDict
from langgraph.graph import StateGraph, END
from langchain_groq import ChatGroq

class StatoDieta(TypedDict):
    query: str
    profilo: str
    risposta_bozza: str
    critica: str
    iterazioni: int

llm = ChatGroq(model_name="llama3-70b-8192", temperature=0.3)

def generatore(stato: StatoDieta):
    prompt = f"Utente: {stato['query']}\nProfilo: {stato['profilo']}\nCritica precedente: {stato['critica']}\nGenera un pasto."
    risposta = llm.invoke(prompt).content
    return {"risposta_bozza": risposta, "iterazioni": stato["iterazioni"] + 1}

def verificatore(stato: StatoDieta):
    prompt = f"Verifica se questa dieta rispetta il profilo '{stato['profilo']}'. Rispondi solo 'PASS' se è perfetta, altrimenti scrivi gli errori.\nDieta: {stato['risposta_bozza']}"
    giudizio = llm.invoke(prompt).content
    return {"critica": giudizio}

def decisore(stato: StatoDieta):
    if "PASS" in stato["critica"].upper() or stato["iterazioni"] >= 3:
        return "fine"
    return "correggi"

# Costruzione del Grafo Ciclico
workflow = StateGraph(StatoDieta)
workflow.add_node("Genera", generatore)
workflow.add_node("Verifica", verificatore)

workflow.set_entry_point("Genera")
workflow.add_edge("Genera", "Verifica")
workflow.add_conditional_edges("Verifica", decisore, {"fine": END, "correggi": "Genera"})

app_grafo = workflow.compile()