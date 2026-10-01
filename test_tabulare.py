import os
import json
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain_groq import ChatGroq

def carica_dataframe_da_json():
    """Converte il dataset JSON del CREA in un DataFrame Pandas usabile dall'agente."""
    file_path = "crea_food_composition_tables.json" if os.path.exists("crea_food_composition_tables.json") else "data/crea_food_composition_tables.json"
    righe = []
    
    if os.path.exists(file_path):
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip(): continue
                try:
                    item = json.loads(line)
                    nome = item.get("name", "")
                    macros = item.get("macro_nutrients", [])
                    
                    def get_m(chiave):
                        for m in macros:
                            if m.get("description") == chiave:
                                try: return float(m.get("value", 0))
                                except: return 0.0
                        return 0.0
                    
                    righe.append({
                        "name": nome,
                        "kcal": get_m("energy_kcal"),
                        "carb": get_m("available_carbohydrates"),
                        "prot": get_m("proteins"),
                        "grassi": get_m("lipids")
                    })
                except: continue
                
    return pd.DataFrame(righe)

def interroga_database_tabulare(query_utente, api_key):
    df = carica_dataframe_da_json()
    
    llm_tabulare = ChatGroq(groq_api_key=api_key, model_name="qwen/qwen3.8-27b", temperature=0)
    
    agente_pandas = create_pandas_dataframe_agent(
        llm_tabulare, 
        df, 
        verbose=True, 
        allow_dangerous_code=True
    )
    
    try:
        risposta = agente_pandas.invoke(query_utente)
        return risposta["output"]
    except Exception as e:
        return f"Errore nell'analisi tabulare: {str(e)}"

def testa_agente_tabulare():
    API_KEY = "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"
    
    query_analitiche = [
        "Qual è l'alimento con il valore più alto di Proteine in assoluto nel database?",
        "Calcola la media delle calorie per tutti gli alimenti nel database."
    ]
    
    print("\n" + "="*60)
    print("INIZIO TEST RAG TABULARE (PANDAS AGENT)")
    print("="*60)
    
    for query in query_analitiche:
        print(f"\n[DOMANDA]: {query}")
        print("Elaborazione in corso (Qwen 27b sta scrivendo codice Python sul JSON)...")
        
        risposta = interroga_database_tabulare(query, API_KEY)
        print(f"[RISPOSTA]: {risposta}\n")
        print("-" * 60)

if __name__ == "__main__":
    testa_agente_tabulare()