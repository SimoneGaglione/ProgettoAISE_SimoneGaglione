import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# Rilevamento dinamico della cartella data
DATA_DIR = os.path.join(BASE_DIR, "data") if os.path.exists(os.path.join(BASE_DIR, "data")) else BASE_DIR

VECTORDB_DIR = os.path.join(BASE_DIR, "vectorstore_ita")
FILE_ALIMENTI = os.path.join(DATA_DIR, "crea_food_composition_tables.json")
FILE_RICETTE = os.path.join(DATA_DIR, "crea_recipes.json")
BM25_FILE = os.path.join(BASE_DIR, "bm25_retriever.pkl")