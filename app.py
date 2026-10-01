import streamlit as st
import os
import base64
import pandas as pd
import altair as alt
import requests
from rag_engine import carica_alimenti_dict
from pdf_generator import crea_pdf_dieta

# --- CONFIGURAZIONE PAGINA ---
st.set_page_config(
    page_title="Assistente Nutrizionale AI", 
    page_icon="🥗", 
    layout="centered",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .stChatMessage { border-radius: 15px; padding: 10px; margin-bottom: 10px; }
    .stButton>button { width: 100%; border-radius: 8px; border: 1px solid #4CAF50; color: #4CAF50; }
    .stButton>button:hover { background-color: #4CAF50; color: white; }
    </style>
""", unsafe_allow_html=True)

st.title("🥗 Assistente Nutrizionale RAG")
st.caption("Carica la tua dieta, imposta i tuoi obiettivi e calcola le sostituzioni in tempo reale.")

# --- SIDEBAR ---
with st.sidebar:
    st.header("⚙️ Pannello di Controllo")
    
    api_key_input = st.session_state.get("groq_key", "")
    gemini_key_input = st.session_state.get("gemini_key", "")
    
    with st.expander("🔑 Impostazioni API", expanded=not bool(api_key_input)):
        api_key_input = st.text_input("Groq API Key", value=api_key_input, type="password")
        gemini_key_input = st.text_input("Gemini API Key", value=gemini_key_input, type="password")
        st.session_state["groq_key"] = api_key_input
        st.session_state["gemini_key"] = gemini_key_input

    with st.expander("👤 Profilo Utente", expanded=True):
        intolleranze = st.multiselect("Intolleranze o Diete:", ["Senza Glutine", "Senza Lattosio", "Vegano", "Vegetariano"])
        obiettivo = st.selectbox("Obiettivo:", ["Mantenimento", "Perdita Peso", "Ipertrofia"])
        profilo_testuale = f"Intolleranze/Dieta: {', '.join(intolleranze) if intolleranze else 'Nessuna'}. Obiettivo: {obiettivo}."

    with st.expander("📸 Analisi Dieta (Foto)", expanded=True):
        foto_dieta = st.file_uploader("Allega la scheda", type=["png", "jpg", "jpeg"], label_visibility="collapsed")
        base64_img = None
        img_mime = "image/jpeg"
        
        if foto_dieta:
            base64_img = base64.b64encode(foto_dieta.read()).decode("utf-8")
            img_mime = "image/png" if foto_dieta.name.lower().endswith("png") else "image/jpeg"
            st.image(foto_dieta, caption="Dieta acquisita", use_container_width=True)
            st.success("Immagine pronta per l'analisi.")

    st.divider()
    if st.button("🗑️ Nuova Conversazione"):
        st.session_state.messages = []
        st.rerun()
        
    if "messages" in st.session_state and len(st.session_state.messages) > 0:
        pdf_bytes = crea_pdf_dieta(st.session_state.messages)
        st.download_button("📄 Esporta in PDF", data=pdf_bytes, file_name="piano_modificato.pdf", mime="application/pdf")

# --- GESTIONE CHATBOT E API ---
if not api_key_input:
    st.info("👋 Benvenuto! Inserisci la tua chiave API di Groq nel pannello laterale per iniziare.")
else:
    @st.cache_resource
    def inizializza_database():
        # Il frontend carica solo il JSON per disegnare i grafici, niente AI!
        return carica_alimenti_dict()

    try:
        alimenti_db = inizializza_database()

        if "messages" not in st.session_state:
            st.session_state.messages = []

        if len(st.session_state.messages) == 0:
            st.markdown("*Nessun messaggio. Fai una domanda per calcolare una sostituzione o chiedere una ricetta!*")

        # --- RENDERING MESSAGGI & DATA VISUALIZATION ---
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                
                alimenti = message.get("alimenti_grafico", [])
                if alimenti:
                    with st.expander("📊 Analisi e Differenze Macronutrienti (Valori per 100g)"):
                        if len(alimenti) == 2:
                            al_1, al_2 = alimenti[0], alimenti[1]
                            m1, m2 = alimenti_db[al_1], alimenti_db[al_2]
                            
                            st.markdown(f"**Confronto diretto: {al_1.title()} ➔ {al_2.title()}**")
                            
                            col1, col2, col3 = st.columns(3)
                            col1.metric("Proteine", f"{m2['prot']}g", f"{(m2['prot'] - m1['prot']):.1f}g", delta_color="off")
                            col2.metric("Carboidrati", f"{m2['carb']}g", f"{(m2['carb'] - m1['carb']):.1f}g", delta_color="off")
                            col3.metric("Grassi", f"{m2['grassi']}g", f"{(m2['grassi'] - m1['grassi']):.1f}g", delta_color="off")
                            
                            df_melted = pd.DataFrame({
                                "Nutriente": ["Proteine", "Carboidrati", "Grassi", "Proteine", "Carboidrati", "Grassi"],
                                "Alimento": [al_1.title()]*3 + [al_2.title()]*3,
                                "Grammi": [m1["prot"], m1["carb"], m1["grassi"], m2["prot"], m2["carb"], m2["grassi"]]
                            })
                            
                            grafico = alt.Chart(df_melted).mark_bar().encode(
                                x=alt.X("Nutriente:N", title=None, sort=["Proteine", "Carboidrati", "Grassi"]),
                                y=alt.Y("Grammi:Q", title="Quantità (g)"),
                                color=alt.Color("Alimento:N", legend=alt.Legend(title=None, orient="bottom")),
                                xOffset="Alimento:N" 
                            )
                            st.altair_chart(grafico, use_container_width=True)
                            
                        elif len(alimenti) == 1:
                            al_1 = alimenti[0]
                            m1 = alimenti_db[al_1]
                            st.markdown(f"**Profilo Nutrizionale: {al_1.title()}**")
                            
                            col1, col2, col3 = st.columns(3)
                            col1.metric("Proteine", f"{m1['prot']}g")
                            col2.metric("Carboidrati", f"{m1['carb']}g")
                            col3.metric("Grassi", f"{m1['grassi']}g")
                            
                            df_melted = pd.DataFrame({
                                "Nutriente": ["Proteine", "Carboidrati", "Grassi"],
                                "Alimento": [al_1.title()]*3,
                                "Grammi": [m1["prot"], m1["carb"], m1["grassi"]]
                            })
                            
                            grafico = alt.Chart(df_melted).mark_bar().encode(
                                x=alt.X("Nutriente:N", title=None, sort=["Proteine", "Carboidrati", "Grassi"]),
                                y=alt.Y("Grammi:Q", title="Quantità (g)"),
                                color=alt.Color("Alimento:N", legend=alt.Legend(title=None, orient="bottom"))
                            )
                            st.altair_chart(grafico, use_container_width=True)
                            
        if prompt := st.chat_input("Scrivi qui la tua richiesta... (es. Con cosa sostituisco 80g di pasta?)"):
            messaggio_display = f"📸 *[Immagine allegata]*\n\n{prompt}" if base64_img else prompt
            st.session_state.messages.append({"role": "user", "content": messaggio_display})
            
            with st.chat_message("user"):
                st.markdown(messaggio_display)

            with st.chat_message("assistant"):
                with st.spinner("Connessione al microservizio in corso..."):
                    storico_chat = "\n".join([f"{msg['role']}: {msg['content']}" for msg in st.session_state.messages[:-1]])
                    
                    parole_chiave_analitiche = ["media", "massimo", "minimo", "totale", "quanti alimenti", "statistiche", "minor", "maggior", "più alto", "più basso", "meno"]
                    is_tab = any(parola in prompt.lower() for parola in parole_chiave_analitiche)
                    if is_tab:
                        st.info("📊 Attivazione Agente Tabulare via API...")

                    payload = {
                        "user_query": prompt,
                        "profilo_utente": profilo_testuale,
                        "storico": storico_chat,
                        "immagine_b64": base64_img,
                        "mime_type": img_mime,
                        "groq_key": api_key_input,
                        "gemini_key": gemini_key_input,
                        "is_tabular": is_tab
                    }
                    
                    try:
                        API_URL = os.getenv("API_URL", "http://backend:8000/api/v1/chat")
                        response = requests.post(API_URL, json=payload)
                        response.raise_for_status()
                        dati = response.json()
                        
                        risposta = dati["risposta"]
                        alimenti_trovati = dati.get("alimenti_estratti", [])
                        st.markdown(risposta)
                        
                    except requests.exceptions.ConnectionError:
                        st.error("Errore: Impossibile raggiungere il Backend. Verifica che il container 'backend' sia attivo.")
                        st.stop()
                    except Exception as e:
                        st.error(f"Errore API: {e}")
                        st.stop()
            
            # MEMORIA DEL GRAFICO
            if len(alimenti_trovati) == 1 and len(st.session_state.messages) >= 2:
                messaggio_precedente = st.session_state.messages[-2] 
                vecchi_alimenti = messaggio_precedente.get("alimenti_grafico", [])
                for vecchio in vecchi_alimenti:
                    if vecchio not in alimenti_trovati:
                        alimenti_trovati.append(vecchio)
                        break 
            
            st.session_state.messages.append({
                "role": "assistant", 
                "content": risposta,
                "alimenti_grafico": alimenti_trovati
            })
            
            st.rerun()

    except Exception as e:
        st.error(f"Errore di sistema: {e}")