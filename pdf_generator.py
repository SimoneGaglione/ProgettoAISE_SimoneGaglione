from fpdf import FPDF

def crea_pdf_dieta(storico_chat):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)
    
    # Titolo del documento
    pdf.set_font("Helvetica", style="B", size=16)
    pdf.cell(0, 10, "Resoconto Sostituzioni Nutrizionali", new_x="LMARGIN", new_y="NEXT", align="C")
    pdf.ln(8)
    
    # Scrittura della chat
    for msg in storico_chat:
        if msg["role"] == "system": 
            continue
            
        ruolo = "Tu" if msg["role"] == "user" else "Nutrizionista"
        
        # Sostituisce caratteri non supportati (come le emoji) con un punto interrogativo per evitare crash
        testo_pulito = msg["content"].replace('\u2019', "'").encode('latin-1', 'replace').decode('latin-1')
        
        # Intestazione del ruolo (Grassetto)
        pdf.set_font("Helvetica", style="B", size=11)
        pdf.multi_cell(0, 6, text=f"{ruolo}:", new_x="LMARGIN", new_y="NEXT")
        
        # Testo del messaggio (Normale)
        pdf.set_font("Helvetica", size=11)
        pdf.multi_cell(0, 6, text=testo_pulito, new_x="LMARGIN", new_y="NEXT")
        pdf.ln(5)
        
    return bytes(pdf.output())