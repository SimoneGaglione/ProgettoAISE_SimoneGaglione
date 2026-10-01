FROM python:3.11-slim

WORKDIR /app

# Usa i nuovi nomi dei pacchetti (libgl1 e libglib2.0-0) per le dipendenze grafiche
RUN apt-get update && apt-get install -y build-essential libgl1 libglib2.0-0 && rm -rf /var/lib/apt/lists/*

# Installa PRIMA torch e torchvision in versione CPU per evitare il timeout di Docker
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Poi copia e installa il resto dei pacchetti
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0"]