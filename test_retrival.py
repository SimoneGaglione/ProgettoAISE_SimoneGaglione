from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.retrievers import BM25Retriever
from sentence_transformers import CrossEncoder 
from ingestion import processa_json_crea 

VECTORDB_DIR = "data/vectorstore_ita"

def reciprocal_rank_fusion(risultati_bm25, risultati_chroma, k=60):
    """
    Implementazione manuale dell'algoritmo Reciprocal Rank Fusion (RRF).
    Fonde matematicamente i risultati della ricerca lessicale e semantica.
    """
    rrf_scores = {}
    
    # 1. Assegna i punteggi per i risultati Lessicali (BM25)
    for rank, doc in enumerate(risultati_bm25, 1):
        testo = doc.page_content
        if testo not in rrf_scores:
            rrf_scores[testo] = {"score": 0.0, "doc": doc}
        rrf_scores[testo]["score"] += 1.0 / (rank + k)
        
    # 2. Assegna i punteggi per i risultati Semantici (ChromaDB)
    for rank, doc in enumerate(risultati_chroma, 1):
        testo = doc.page_content
        if testo not in rrf_scores:
            rrf_scores[testo] = {"score": 0.0, "doc": doc}
        rrf_scores[testo]["score"] += 1.0 / (rank + k)
        
    # 3. Ordina i documenti in base allo score RRF finale
    documenti_fusi = sorted(rrf_scores.values(), key=lambda x: x["score"], reverse=True)
    return [item["doc"] for item in documenti_fusi]

def testa_retriever_ibrido_con_rerank():
    print("1. Estrazione dei documenti per il motore lessicale (BM25)...")
    docs = processa_json_crea()
    
    print("2. Creazione del Retriever Lessicale (BM25)...")
    bm25_retriever = BM25Retriever.from_documents(docs)
    bm25_retriever.k = 10  # Estraiamo un ampio set grezzo per mitigare il 'Lost in the Middle'
    
    print("3. Connessione al Retriever Semantico (ChromaDB)...")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    vectorstore = Chroma(
        persist_directory=VECTORDB_DIR,
        embedding_function=embeddings
    )
    chroma_retriever = vectorstore.as_retriever(search_kwargs={"k": 10})
    
    print("4. Inizializzazione del Cross-Encoder per il Reranking profondo...")
    reranker = CrossEncoder('cross-encoder/mmarco-mMiniLMv2-L12-H384-v1')
    
    query_di_prova = [
        "Voglio i valori nutrizionali del pollo",
        "Come si prepara la lasagna?",
        "Cerco un carboidrato per sostituire la pasta"
    ]
    
    print("\n" + "="*60)
    print("INIZIO TEST HYBRID SEARCH + RERANKING (RRF + CROSS-ENCODER)")
    print("="*60)
    
    for query in query_di_prova:
        print(f"\n[QUERY UTENTE]: '{query}'")
        
        # Fase 1: Recupero ibrido con RRF
        risultati_bm25 = bm25_retriever.invoke(query)
        risultati_chroma = chroma_retriever.invoke(query)
        risultati_fusi = reciprocal_rank_fusion(risultati_bm25, risultati_chroma)
        
        # Fase 2: Reranking profondo per eliminare il rumore e ordinare semanticamente
        pairs = [[query, doc.page_content] for doc in risultati_fusi]
        scores = reranker.predict(pairs)
        
        doc_score_pairs = list(zip(risultati_fusi, scores))
        doc_score_pairs.sort(key=lambda x: x[1], reverse=True)
        
        print("\n--- TOP 3 RISULTATI DOPO IL RERANKING ---")
        for i, (doc, score) in enumerate(doc_score_pairs[:3], 1):
            tipo = doc.metadata.get('tipo', 'sconosciuto').upper()
            nome_o_codice = doc.metadata.get('nome', doc.metadata.get('codice', 'N/A'))
            print(f"  Posizione {i} | Score Cross-Encoder: {score:.4f} | Fonte ({tipo}: {nome_o_codice})")
            contenuto_troncato = doc.page_content.replace('\n', ' ')[:120]
            print(f"  Testo: {contenuto_troncato}...\n")

if __name__ == "__main__":
    testa_retriever_ibrido_con_rerank()