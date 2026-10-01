import json
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# IMPORTIAMO LA FUNZIONE DAL NOSTRO MOTORE PRINCIPALE!
from rag_engine import get_working_model 

class LLMJudge:
    def __init__(self, api_key):
        modello_valido = get_working_model(api_key)
        
        self.llm = ChatGroq(
            groq_api_key=api_key, 
            model_name=modello_valido, 
            temperature=0.0
        )
        self.llm_json = self.llm.bind(response_format={"type": "json_object"})
        
        self.judge_prompt = ChatPromptTemplate.from_template("""
        Sei un Giudice esperto in Intelligenza Artificiale. Valuta la seguente interazione di un sistema RAG (Retrieval-Augmented Generation).
        
        Domanda Utente: {query}
        Risposta del Sistema RAG: {risposta}
        
        Devi valutare la risposta basandoti su due criteri:
        1. Factual Correctness (Correttezza): Il sistema ha inventato dati o è rimasto fedele al suo ruolo?
        2. Tone & Style: Il tono è cordiale, professionale e privo di ragionamenti matematici visibili?
        
        RISPONDI SOLO CON UN JSON VALIDO in questo formato:
        {{
            "score": [voto da 1 a 5, dove 5 è perfetto],
            "reasoning": "Breve spiegazione del voto"
        }}
        """)
        
        self.chain = self.judge_prompt | self.llm_json | StrOutputParser()

    def valuta_risposta(self, query, risposta):
        print("Valutazione LLM-as-a-Judge in corso...")
        try:
            risultato_json = self.chain.invoke({"query": query, "risposta": risposta})
            return json.loads(risultato_json)
        except Exception as e:
            return {"score": 0, "reasoning": f"Errore del giudice: {str(e)}"}

if __name__ == "__main__":
    # INSERISCI DI NUOVO QUI LA TUA VERA CHIAVE API GROQ (gsk_...)
    TEST_API_KEY = "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"
    
    if TEST_API_KEY != "LA_TUA_CHIAVE_API_GROQ_QUI":
        giudice = LLMJudge(TEST_API_KEY)
        
        query_test = "Quante calorie ha il riso?"
        risposta_test = "Non ho il dato del riso semplice, ma la farina di riso ha 332 calorie."
        
        valutazione = giudice.valuta_risposta(query_test, risposta_test)
        print("\n--- RISULTATO LLM-AS-A-JUDGE ---")
        print(f"Punteggio: {valutazione['score']} / 5")
        print(f"Motivazione: {valutazione['reasoning']}")
    else:
        print("Inserisci la tua chiave API per eseguire il giudice.")