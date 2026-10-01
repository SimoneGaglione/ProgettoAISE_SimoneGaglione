import os
import re
import requests
import difflib
import json
import base64
import pandas as pd
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain_chroma import Chroma
from langchain_core.messages import HumanMessage
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from sentence_transformers import CrossEncoder 
from groq import Groq
from ingestion import processa_json_crea 

# Gestione dinamica dei percorsi
VECTORDB_DIR = "./vectorstore_ita" if os.path.exists("./vectorstore_ita") else "data/vectorstore_ita"
FILE_ALIMENTI = "crea_food_composition_tables.json" if os.path.exists("crea_food_composition_tables.json") else "data/crea_food_composition_tables.json"
FILE_RICETTE = "crea_recipes.json" if os.path.exists("crea_recipes.json") else "data/crea_recipes.json"

def carica_alimenti_dict():
    alimenti = {}
    if os.path.exists(FILE_ALIMENTI):
        with open(FILE_ALIMENTI, "r", encoding="utf-8") as f:
            for linea in f:
                if not linea.strip(): continue
                try:
                    item = json.loads(linea)
                    nome = item.get("name", "").lower().strip()
                    macros = item.get("macro_nutrients", [])
                    
                    def get_m(chiave):
                        for m in macros:
                            if m.get("description") == chiave:
                                try: return float(m.get("value", 0))
                                except: return 0.0
                        return 0.0
                    
                    alimenti[nome] = {
                        "kcal": get_m("energy_kcal"),
                        "carb": get_m("available_carbohydrates"),
                        "prot": get_m("proteins"),
                        "grassi": get_m("lipids")
                    }
                except: continue
    return alimenti

def carica_ricette_list():
    ricette = []
    if os.path.exists(FILE_RICETTE):
        with open(FILE_RICETTE, "r", encoding="utf-8") as f:
            for linea in f:
                if not linea.strip(): continue
                try:
                    ricette.append(json.loads(linea))
                except: continue
    return ricette

# --- CLASSE RAG AVANZATA ---
class ChatNutriRAG:
    def __init__(self, groq_api_key, gemini_api_key=None):
        print("Inizializzazione del Modello Conversazionale Nutrizionale...")
        self.api_key = groq_api_key 
        self.gemini_api_key = gemini_api_key
        self.alimenti_db = carica_alimenti_dict()
        self.ricette_db = carica_ricette_list()
        self.ultimi_alimenti = []
      

        modello_scelto = "qwen/qwen3.8-27b"
        print(f"Modello Groq bloccato su: {modello_scelto}")
        
        # 1. Inizializziamo il modello PRINCIPALE 
        self.llm = ChatGroq(
            groq_api_key=self.api_key,
            model_name=modello_scelto,
            temperature=0.0
        )
        
        # 2. Inizializziamo il modello JSON 
        self.llm_json = ChatGroq(
            groq_api_key=self.api_key,
            model_name=modello_scelto, 
            temperature=0.0
        )
        
        # INIZIALIZZAZIONE RETRIEVERS 
        docs = processa_json_crea()
        self.bm25_retriever = BM25Retriever.from_documents(docs)
        self.bm25_retriever.k = 10
        
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
        self.vectorstore = Chroma(persist_directory=VECTORDB_DIR, embedding_function=embeddings)
        self.chroma_retriever = self.vectorstore.as_retriever(search_kwargs={"k": 10})
        
        # INIZIALIZZAZIONE RERANKER 
        self.reranker = CrossEncoder('cross-encoder/mmarco-mMiniLMv2-L12-H384-v1')
        # PROMPT PRINCIPALE
        # PROMPT PRINCIPALE
        self.prompt = ChatPromptTemplate.from_template("""
        Sei un brillante e rigoroso Assistente Nutrizionista.

        STORICO DELLA CONVERSAZIONE:
        {storico}
        
        PROFILO E PREFERENZE UTENTE:
        {profilo_utente}
        
        DATI UFFICIALI CREA E CALCOLI DETERMINISTICI (FONTE ASSOLUTA):
        {dati_deterministici}
        
        CONTESTO RAG:
        {contesto_rag}
        
        DOMANDA DELL'UTENTE:
        {question}
        
        REGOLE INVALICABILI:
        1. RISPETTO DEL PROFILO: Leggi il "PROFILO UTENTE". Se c'è scritto VEGANO, è ASSOLUTAMENTE VIETATO proporre latticini, uova, carne o pesce. Se è SENZA LATTOSIO, zero formaggi tradizionali. Usa alternative 100% vegetali.
        2. NUMERI DETERMINISTICI: Se nei "DATI UFFICIALI" trovi scritto un calcolo (es. "servono X g"), DEVI usare QUEL NUMERO ESATTO. È severamente vietato ricalcolare le proporzioni da solo.
        3. STILE: Rispondi in modo discorsivo. Niente elenchi puntati lunghi. Non mostrare i calcoli matematici.
        4. GESTIONE AMBIGUITÀ: Se un alimento richiesto è generico e devi chiedere all'utente di specificare, offri SOLO 2 o 3 esempi classici, concreti e tradizionali.
        5. CALCOLI MULTIPLI E TOTALI: Se l'utente chiede la somma di più alimenti, calcolala in modo discorsivo basandoti sulle quantità indicate nella foto.
        6. BUONSENSO : Se l'utente chiede un alimento specifico non presente nel tuo database , non cercare soluzioni autonomamente ma chiedi sempre un chiarimento.
        RISPOSTA:
        """)
        
        
        self.safety_prompt = ChatPromptTemplate.from_template("""
        Sei un revisore editoriale. Verifica questa risposta:
        
        Risposta generata: {risposta}
        Dati reali obbligatori: {dati_deterministici}
        Profilo utente: {profilo_utente}
        
        REGOLE DI REVISIONE:
        1. Se la risposta propone alimenti animali a un Vegano o lattosio a un intollerante, CAMBIA L'ALIMENTO con un'alternativa sicura.
        2. Se i grammi non combaciano con "Dati reali obbligatori", correggili.
        3. Nascondi le formule matematiche.
        
        RISPONDI SOLO CON IL SEGUENTE JSON VALIDO:
        {{
            "risposta_finale": "scrivi qui la tua risposta finale revisionata"
        }}
        """)
        
        self.output_parser = StrOutputParser()
        self.chain = self.prompt | self.llm | self.output_parser
        self.safety_chain = self.safety_prompt | self.llm_json | StrOutputParser()

    def reciprocal_rank_fusion(self, risultati_bm25, risultati_chroma, k=60):
        rrf_scores = {}
        for rank, doc in enumerate(risultati_bm25, 1):
            testo = doc.page_content
            if testo not in rrf_scores: rrf_scores[testo] = {"score": 0.0, "doc": doc}
            rrf_scores[testo]["score"] += 1.0 / (rank + k)
            
        for rank, doc in enumerate(risultati_chroma, 1):
            testo = doc.page_content
            if testo not in rrf_scores: rrf_scores[testo] = {"score": 0.0, "doc": doc}
            rrf_scores[testo]["score"] += 1.0 / (rank + k)
            
        return [item["doc"] for item in sorted(rrf_scores.values(), key=lambda x: x["score"], reverse=True)]

    def recupero_ibrido_rerank(self, query, top_k=7):
        docs_bm25 = self.bm25_retriever.invoke(query)
        docs_chroma = self.chroma_retriever.invoke(query)
        docs_fusi = self.reciprocal_rank_fusion(docs_bm25, docs_chroma)
        
        if not docs_fusi: return ""
        pairs = [[query, doc.page_content] for doc in docs_fusi]
        scores = self.reranker.predict(pairs)
        
        doc_score_pairs = list(zip(docs_fusi, scores))
        doc_score_pairs.sort(key=lambda x: x[1], reverse=True)
        
        migliori_docs = [doc.page_content for doc, score in doc_score_pairs[:top_k]]
        return "\n\n".join(migliori_docs)

    def estrai_entita_llm(self, user_query, storico="", testo_visione=""):
        estrazione_prompt = ChatPromptTemplate.from_template("""
        Sei un estrattore dati JSON rigoroso.
        
        RICHIESTA ATTUALE: "{query}"
        TESTO FOTO: "{visione}"
        STORICO CONVERSAZIONE: "{storico}"
        
        REGOLE:
        1. Identifica l'alimento di partenza (alimento_orig), l'alimento sostitutivo (alimento_target) e la grammatura (grammi_orig).
        2. Determina quale macronutriente pareggiare (macro_focus): scegli tra "prot", "carb", "grassi" o "kcal". Se l'utente menziona proteine, usa "prot". Di default sui cereali usa "carb", sulla carne/pesce usa "prot".
        3. CRITICO: Se la "RICHIESTA ATTUALE" è solo il nome di un alimento , guarda lo "STORICO CONVERSAZIONE". Cerca la domanda precedente dell'utente per recuperare i grammi e l'altro alimento mancante.
        
        Restituisci SOLO un JSON:
        {{
           "grammi_orig": 100.0,
           "alimento_orig": "stringa o null",
           "alimento_target": "stringa o null",
           "macro_focus": "prot" | "carb" | "grassi" | "kcal"
        }}
        """)
        chain_estrazione = estrazione_prompt | self.llm_json | StrOutputParser()
        
        try:
            raw_out = chain_estrazione.invoke({"query": user_query, "visione": testo_visione, "storico": storico})
            clean_out = re.sub(r'<think>.*?(?:</think>|$)', '', raw_out, flags=re.DOTALL).strip()
            clean_out = clean_out.replace('```json', '').replace('```', '').strip()
            match_json = re.search(r'\{.*?\}', clean_out, flags=re.DOTALL)
            if match_json:
                clean_out = match_json.group(0)
            return json.loads(clean_out)
        except Exception:
            return {"grammi_orig": 100.0, "alimento_orig": None, "alimento_target": None, "macro_focus": "carb"}

    def trova_chiave_alimento(self, query_parola):
        if not query_parola: return None, []
        
        # 1. Pulizia stringa
        query_pulita = re.sub(r'[^\w\s]', '', query_parola.lower()).strip()
        
       
        q_words = set(w for w in query_pulita.split() if len(w) > 2)
        chiavi_db = list(self.alimenti_db.keys())

        
        for k in chiavi_db:
            if query_pulita == re.sub(r'[^\w\s]', '', k.lower()).strip():
                return k, []

        candidati = []
        
        if q_words:
            for k in chiavi_db:
                k_clean = re.sub(r'[^\w\s]', '', k.lower())
                k_words = set(w for w in k_clean.split() if len(w) > 2)
                
                if not k_words: continue
                
                intersezione = q_words.intersection(k_words)
                
                if intersezione:
                    
                    copertura_query = len(intersezione) / len(q_words)
                    
                    
                    if copertura_query >= 0.6:
                        candidati.append((copertura_query, k))

        
        if candidati:
           
            parola_principale = query_pulita.split()[0] if query_pulita else ""
            
            
            candidati.sort(reverse=True, key=lambda x: (
                x[0], 
                x[1].startswith(parola_principale), 
                -len(x[1])
            ))
            
            
            if candidati[0][0] == 1.0:
                perfetti = [c for c in candidati if c[0] == 1.0]
                
                if len(perfetti) == 1:
                    return perfetti[0][1], []
                 
                else:
                    return None, [c[1] for c in perfetti[:3]]
            
           
            return None, [c[1] for c in candidati[:3]]

        # CASO 3: Rete di Sicurezza per Errori di Battitura 
        match_simili = difflib.get_close_matches(query_pulita, chiavi_db, n=3, cutoff=0.6)
        if match_simili:
            return None, match_simili

        # Nessun alimento trovato
        return None, []

    def calcola_sostituzione_libera(self, user_query, storico="", testo_visione=""):
        self.ultimi_alimenti = []
        entita = self.estrai_entita_llm(user_query, storico, testo_visione)
        orig_parola = entita.get("alimento_orig")
        target_parola = entita.get("alimento_target")
        grammi_orig = float(entita.get("grammi_orig") or 100.0)
        macro = entita.get("macro_focus") or "carb"
        
        nomi_macro = {"prot": "proteine", "carb": "carboidrati", "grassi": "grassi", "kcal": "calorie"}

        key_orig = None
        key_target = None

        if orig_parola:
            key_orig, opzioni_orig = self.trova_chiave_alimento(orig_parola)
            if not key_orig:
                if opzioni_orig:
                    lista_orig = "\n".join([f"- **{opt.title()}**" for opt in opzioni_orig])
                    return f"ATTENZIONE: L'alimento '{orig_parola}' è ambiguo. Chiedi di specificare tra:\n{lista_orig}"
                else:
                    return f"ISTRUZIONE: L'alimento '{orig_parola}' non esiste nel database ufficiale CREA. Informa l'utente e chiedi di inserire un alimento diverso."
        
        if target_parola:
            key_target, opzioni_target = self.trova_chiave_alimento(target_parola)
            if not key_target:
                if opzioni_target:
                    lista_target = "\n".join([f"- **{opt.title()}**" for opt in opzioni_target])
                    return f"ATTENZIONE: L'alimento '{target_parola}' è ambiguo. Chiedi di specificare tra:\n{lista_target}"
                else:
                    return f"ISTRUZIONE: L'alimento '{target_parola}' non è presente nei dati CREA. Informa l'utente che non puoi calcolare la sostituzione e suggerisci tu un paio di alternative valide."

        if key_orig and key_target and key_orig in self.alimenti_db and key_target in self.alimenti_db:
            val_orig_100 = self.alimenti_db[key_orig][macro]
            val_target_100 = self.alimenti_db[key_target][macro]
            quota_totale = (val_orig_100 * grammi_orig) / 100.0
            
            if val_target_100 > 0:
                grammi_target = (quota_totale * 100.0) / val_target_100
                self.ultimi_alimenti = [key_orig, key_target]
                return (f"Calcolo deterministico:\n"
                        f"Per eguagliare i {quota_totale:.1f}g di {nomi_macro.get(macro, 'nutrienti')} in {grammi_orig}g di {key_orig}, "
                        f"servono {grammi_target:.1f}g di {key_target}.")
        elif key_orig and not target_parola:
            val_orig_100 = self.alimenti_db[key_orig][macro]
            quota_totale = (val_orig_100 * grammi_orig) / 100.0
            self.ultimi_alimenti = [key_orig]
            return (f"ISTRUZIONE PER IL LLM: L'utente deve assumere {quota_totale:.1f}g di {nomi_macro.get(macro, 'nutrienti')} (equivalenti a {grammi_orig}g di {key_orig}). "
                    f"Proponi 2 o 3 alternative adeguate dal database CREA.")
                        
        return "Nessun calcolo deterministico automatico applicabile."

    def gestisci_chat(self, user_query, profilo_utente="Nessuna preferenza specifica", storico="", immagine_b64=None, mime_type="image/jpeg"):
        contesto_visivo = ""
        testo_estratto_vision = ""
        
       # --- 1. INTEGRAZIONE MODELLO VISION 
        if immagine_b64:
            if not self.gemini_api_key:
                return "📸 Ho ricevuto l'immagine, ma manca la chiave API di Gemini. Inseriscila nella barra laterale!"

            print(f"Analisi dell'immagine ({mime_type}) con SDK Ufficiale Gemini in corso...")
            try:
                import io
                from PIL import Image
                from google import genai
                from google.genai import types
                
                
                image_bytes = base64.b64decode(immagine_b64)
                img = Image.open(io.BytesIO(image_bytes))
                
                if img.mode in ('RGBA', 'P'):
                    img = img.convert('RGB')
                    
                img.thumbnail((800, 800))
                
                buffer = io.BytesIO()
                img.save(buffer, format="JPEG", quality=80)
                immagine_compressa_bytes = buffer.getvalue() 
                
                
                client = genai.Client(api_key=self.gemini_api_key)
                
                prompt_visione = f"L'utente mi ha posto questa domanda: '{user_query}'. Leggi la dieta dalla foto allegata. Individua il pasto richiesto dall'utente ed estrai i suoi alimenti con le grammature esatte. Se non è specificato, estrai tutta la giornata."
                
                response = client.models.generate_content(
                    model='gemini-3.6-flash',  
                    contents=[
                        prompt_visione,
                        types.Part.from_bytes(
                            data=immagine_compressa_bytes,
                            mime_type='image/jpeg'
                        )
                    ],
                    config=types.GenerateContentConfig(
                        temperature=0.0
                    )
                )
                
                testo_estratto_vision = response.text
                
                print(f"ESTRATTO DALLA FOTO:\n{testo_estratto_vision}\n")
                contesto_visivo = f"INFORMAZIONI LETTE DALLA FOTO:\n{testo_estratto_vision}\n"
                
            except Exception as e:
                print(f"ERRORE NEL MODELLO VISION SDK: {e}")
                contesto_visivo = "[Errore: Impossibile leggere l'immagine con Gemini]"
        # --- 2. RECUPERO RAG STANDARD ---
        contesto_recuperato = self.recupero_ibrido_rerank(user_query)
        dati_contesto = self.calcola_sostituzione_libera(user_query, storico, testo_estratto_vision)
        contesto_totale = contesto_visivo + "\n" + contesto_recuperato
        
        # --- 3. GENERAZIONE BOZZA ---
        try:
            risposta_grezza = self.chain.invoke({
                "storico": storico,
                "profilo_utente": profilo_utente,
                "dati_deterministici": dati_contesto, 
                "contesto_rag": contesto_totale, 
                "question": user_query
            })
        except Exception as e:
            print(f"Errore nella generazione bozza: {e}")
            return "Si è verificato un errore di connessione con il LLM principale."
        
        risposta_grezza_pulita = re.sub(r'<think>.*?(?:</think>|$)', '', risposta_grezza, flags=re.DOTALL).strip()
       
        
        # --- 4. SAFETY/CRITIQUE PASS ---
        try:
            risposta_sicura_json = self.safety_chain.invoke({
                "risposta": risposta_grezza_pulita,
                "dati_deterministici": dati_contesto,
                "profilo_utente": profilo_utente
            })
            
            risposta_sicura_json = re.sub(r'<think>.*?(?:</think>|$)', '', risposta_sicura_json, flags=re.DOTALL).strip()
            risposta_sicura_json = risposta_sicura_json.replace('```json', '').replace('```', '').strip()
            
            if not risposta_sicura_json:
                risposta_finale = risposta_grezza_pulita
            else:
                try:
                    risposta_dict = json.loads(risposta_sicura_json)
                    risposta_finale = risposta_dict.get("risposta_finale", "")
                except json.JSONDecodeError:
                    # BLOCCO EMERGENZA: Estrae la stringa anche se il JSON è tagliato a metà
                    match_risposta = re.search(r'"risposta_finale"\s*:\s*"([^"]+)', risposta_sicura_json)
                    if match_risposta:
                        risposta_finale = match_risposta.group(1)
                    else:
                        risposta_finale = risposta_grezza_pulita
                
                if not risposta_finale or str(risposta_finale).strip().lower() == "none":
                    risposta_finale = risposta_grezza_pulita
                
        except Exception as e:
            print(f"Errore nel parsing JSON del Safety Layer: {e}")
            risposta_finale = risposta_grezza_pulita
            
        if not risposta_finale or str(risposta_finale).strip().lower() == "none":
            return "Scusa, si è verificata una collisione interna nei dati. Potresti riformulare la richiesta?"
            
        return risposta_finale
    
def interroga_database_tabulare(query_utente, api_key):
    """Usa un agente Pandas per interrogare il JSON CREA convertito in DataFrame."""
    import os
    import json
    
    # 1. Recupero dinamico del JSON
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
                
    df = pd.DataFrame(righe)
    
    # 2. Inizializzazione Agente
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