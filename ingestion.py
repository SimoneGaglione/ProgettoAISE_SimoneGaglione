import json
import os
from langchain_core.documents import Document
from langchain_chroma import Chroma  # <-- Import aggiornato!
from langchain_huggingface import HuggingFaceEmbeddings

# Rilevamento dinamico dei file (funziona sia che tu sia in 'data' sia in 'aise')
# Se il file esiste nella root usa quello, altrimenti cercalo in data/
FILE_ALIMENTI = "crea_food_composition_tables.json" if os.path.exists("crea_food_composition_tables.json") else "data/crea_food_composition_tables.json"
FILE_RICETTE = "crea_recipes.json" if os.path.exists("crea_recipes.json") else "data/crea_recipes.json"
VECTORDB_DIR = "./vectorstore_ita"  # <-- Nuova cartella per il DB in italiano

def estrai_macro(macros, chiave):
    if not isinstance(macros, list): return 0.0
    for m in macros:
        if m.get("description") == chiave:
            try:
                return float(m.get("value", 0))
            except (ValueError, TypeError):
                return 0.0
    return 0.0

def processa_json_crea():
    documents = []
    
    # 1. ALIMENTI
    if os.path.exists(FILE_ALIMENTI):
        with open(FILE_ALIMENTI, "r", encoding="utf-8") as f:
            for linea in f:
                if not linea.strip(): continue
                try:
                    item = json.loads(linea)
                    nome = item.get("name", "Sconosciuto")
                    categoria = item.get("category", "Generico")
                    macros = item.get("macro_nutrients", [])
                    
                    kcal = estrai_macro(macros, "energy_kcal")
                    carb = estrai_macro(macros, "available_carbohydrates")
                    prot = estrai_macro(macros, "proteins")
                    grassi = estrai_macro(macros, "lipids")
                    fibre = estrai_macro(macros, "total_fiber")
                    
                    testo = (
                        f"ALIMENTO: {nome}\n"
                        f"CATEGORIA: {categoria}\n"
                        f"Valori nutrizionali per 100g:\n"
                        f"- Calorie: {kcal} kcal\n"
                        f"- Carboidrati: {carb} g\n"
                        f"- Proteine: {prot} g\n"
                        f"- Grassi: {grassi} g\n"
                        f"- Fibre: {fibre} g\n"
                    )
                    metadati = {"tipo": "alimento", "nome": str(nome)}
                    documents.append(Document(page_content=testo, metadata=metadati))
                except json.JSONDecodeError: continue

    # 2. RICETTE
    if os.path.exists(FILE_RICETTE):
        with open(FILE_RICETTE, "r", encoding="utf-8") as f:
            for linea in f:
                if not linea.strip(): continue
                try:
                    item = json.loads(linea)
                    codice = item.get("food_code", "N/A")
                    porzioni = item.get("portions", "N/A")
                    preparazione = item.get("preparation", "")
                    
                    ingredienti_str = ""
                    for ing in item.get("ingredients", []):
                        nome_ing = ing.get("name", "")
                        quantita = ing.get("quantity", "")
                        if nome_ing:
                            ingredienti_str += f"  * {nome_ing}: {quantita}\n"
                            
                    testo = (
                        f"RICETTA CODICE: {codice}\n"
                        f"PORZIONI: {porzioni}\n"
                        f"INGREDIENTI:\n{ingredienti_str}\n"
                        f"PREPARAZIONE: {preparazione}\n"
                    )
                    metadati = {"tipo": "ricetta", "codice": str(codice)}
                    documents.append(Document(page_content=testo, metadata=metadati))
                except json.JSONDecodeError: continue

    return documents

def build_vector_database():
    docs = processa_json_crea()
    if len(docs) == 0:
        print("ERRORE: 0 documenti estratti.")
        return
        
    print(f"Estratti {len(docs)} documenti pronti per il RAG Italiano.")
    print("Download e inizializzazione del Modello Multilingua (questa volta capirà l'italiano!)...")
    
    # IL SEGRETO È QUI: Modello Multilingua avanzato!
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    
    vectorstore = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        persist_directory=VECTORDB_DIR
    )
    
    print(f"=== VITTORIA! ===\nDatabase ITALIANO popolato con successo in: {VECTORDB_DIR}")

if __name__ == "__main__":
    build_vector_database()