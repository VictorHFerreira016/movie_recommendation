FROM python:3.11-slim

# Evita prompts interativos durante a instalação
ENV DEBIAN_FRONTEND=noninteractive

# Instala dependências do sistema
RUN apt-get update && apt-get install -y \
    build-essential \
    python3-dev \
    libfreetype6-dev \
    libpng-dev \
    libjpeg-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Define diretório de trabalho
WORKDIR /app

# Copia e instala dependências Python
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copia código da aplicação
COPY scripts/ scripts/
COPY main.py .
COPY interface.py .

# Copia dados brutos (necessários para o pipeline)
COPY data/data_raw/ data/data_raw/

# Cria diretórios para dados processados
RUN mkdir -p data/data_processed data/processed

# Define variáveis de ambiente
ENV PYTHONPATH=/app
ENV STREAMLIT_SERVER_PORT=8501
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0
ENV STREAMLIT_SERVER_HEADLESS=true
ENV STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

# Expõe porta do Streamlit
EXPOSE 8501

# Healthcheck
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

# Comando para rodar a aplicação
CMD ["streamlit", "run", "interface.py", "--server.port=8501", "--server.address=0.0.0.0"]