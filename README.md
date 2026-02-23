# Movie Recommendation System

Sistema de recomendação de filmes com busca inteligente e explicações geradas por IA. Utiliza embeddings semânticos (Pinecone) ou TF-IDF para encontrar filmes similares e LLM (Groq) para gerar explicações personalizadas.

## 🛠️ Ferramentas e Tecnologias

* **Linguagem**: Python 3.9+
* **Interface**: Streamlit
* **Embeddings**: Sentence Transformers (all-MiniLM-L6-v2)
* **Vector Database**: Pinecone
* **LLM**: Groq API (Llama 3 70B)
* **Framework IA**: LangChain
* **Busca Fuzzy**: RapidFuzz
* **Validação**: Pydantic
* **Processamento**: Pandas, NumPy, Scikit-learn
* **Containerização**: Docker

## 🎯 Funcionalidades

* Busca inteligente com correção automática de digitação
* Recomendação baseada em conteúdo (gênero, enredo, elenco, diretor)
* Dois métodos de similaridade: TF-IDF ou Embeddings Semânticos
* Explicações geradas por IA sobre por que os filmes são similares
* Interface web interativa e responsiva

## ⚙️ Configuração de Dados

### 1. Download do Dataset

Baixe o dataset TMDB 5000 Movie Dataset do Kaggle:
* **Link:** https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata

### 2. Organização dos Arquivos

Após extrair o arquivo ZIP, organize assim:
```
movie-recommendation/
└── data/
    └── data_raw/
        ├── tmdb_5000_credits.csv
        └── tmdb_5000_movies.csv
```

### 3. Configuração das API Keys

Crie um arquivo `.env` na raiz do projeto:
```env
GROQ_API_KEY=sua_chave_groq_aqui
PINECONE_API_KEY=sua_chave_pinecone_aqui
PINECONE_ENVIRONMENT=us-east-1
```

**Como obter as chaves:**
* **Groq**: https://console.groq.com/
* **Pinecone**: https://www.pinecone.io/

## 🚀 Como Executar (Docker - Recomendado)

### Pré-requisitos

* [Git](https://git-scm.com/)
* [Docker Desktop](https://www.docker.com/products/docker-desktop/)

### Passos

1. **Clone o repositório**
```bash
   git clone https://github.com/VictorHFerreira016/movie-recommendation.git
   cd movie-recommendation
```

2. **Configure o arquivo .env** (veja seção acima)

3. **Build da imagem Docker**
```bash
   docker build -t movie-recommender .
```

4. **Execute o container**
```bash
   docker run -p 8501:8501 --env-file .env movie-recommender
```

5. **Acesse a aplicação**
   
   Abra seu navegador em: **http://localhost:8501**

## 💻 Como Executar (Ambiente Local)

### Pré-requisitos

* Python 3.9 ou superior
* Git

### Passos

1. **Clone o repositório**
```bash
   git clone https://github.com/VictorHFerreira016/movie-recommendation.git
   cd movie-recommendation
```

2. **Crie e ative o ambiente virtual**
   
   *Windows:*
```bash
   python -m venv venv
   .\venv\Scripts\activate
```
   
   *macOS/Linux:*
```bash
   python3 -m venv venv
   source venv/bin/activate
```

3. **Instale as dependências**
```bash
   pip install -r requirements.txt
```

4. **Configure o arquivo .env** (veja seção de configuração)

5. **Execute o pipeline de processamento**
```bash
   python main.py --method embeddings
```
   
   Ou para usar TF-IDF:
```bash
   python main.py --method tfidf
```

6. **Inicie a interface Streamlit**
```bash
   streamlit run interface.py
```

7. **Acesse a aplicação**
   
   O Streamlit abrirá automaticamente em: **http://localhost:8501**

## 📊 Métodos de Recomendação

### TF-IDF (Tradicional)
* Vetorização baseada em frequência de termos
* Rápido e eficiente
* Não requer APIs externas
* Configure: `USE_EMBEDDINGS=False` no .env

### Embeddings Semânticos (IA)
* Compreensão semântica profunda
* Utiliza Sentence Transformers
* Armazenamento vetorial no Pinecone
* Explicações geradas por LLM
* Configure: `USE_EMBEDDINGS=True` no .env

## 📁 Estrutura do Projeto
```
movie-recommendation/
├── Dockerfile                      # Ambiente Docker
├── requirements.txt                # Dependências Python
├── .env                            # Variáveis de ambiente (criar)
├── main.py                         # Pipeline principal
├── interface.py                    # Interface Streamlit
├── data/
│   ├── data_raw/                   # Datasets originais
│   ├── data_processed/             # Dados limpos
│   └── processed/                  # Embeddings e IDs
├── scripts/
│   ├── config.py                   # Configurações (Pydantic)
│   ├── preprocessing.py            # Limpeza de dados
│   ├── embeddings.py               # Geração e Pinecone
│   ├── recommender.py              # Motor de recomendação
│   ├── evaluation.py               # Métricas de avaliação
│   ├── utils.py                    # Utilitários
│   └── visualization.py            # Gráficos e wordclouds
├── notebooks/                      # Jupyter notebooks
└── tests/                          # Testes unitários
```

## 🧪 Testes e Avaliação

Execute os testes:
```bash
pytest tests/
```

Compare métodos (TF-IDF vs Embeddings):
```bash
python scripts/evaluation.py
```

Métricas avaliadas:
* Precision@K
* Recall@K
* MAP (Mean Average Precision)

## 🎨 Visualizações

O projeto inclui:
* Gráficos de similaridade (barras horizontais)
* Word clouds dos filmes recomendados

Gerados automaticamente em `scripts/visualization.py`

## 🔧 Configurações Avançadas

Edite `scripts/config.py` ou use variáveis de ambiente:
```env
GROQ_MODEL=llama3-70b-8192
GROQ_TEMPERATURE=0.7
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
RETRIEVAL_TOP_K=10
SIMILARITY_THRESHOLD=0.7
ENABLE_EXPLANATIONS=True
```

## 🤝 Contribuindo

1. Fork o projeto
2. Crie uma branch: `git checkout -b feature/nova-funcionalidade`
3. Commit: `git commit -m 'Adiciona nova funcionalidade'`
4. Push: `git push origin feature/nova-funcionalidade`
5. Abra um Pull Request

## 📝 Licença

Este projeto está sob a licença especificada em `LICENSE.txt`

## 👤 Autor

**Victor Hugo**

* LinkedIn: [Victor Hugo](https://www.linkedin.com/in/victor-hugo-0a7821300/)
* GitHub: [@VictorHFerreira016](https://github.com/VictorHFerreira016)

## 📚 Referências

* [Sentence Transformers](https://www.sbert.net/)
* [Pinecone Documentation](https://docs.pinecone.io/)
* [LangChain](https://python.langchain.com/)
* [Groq API](https://console.groq.com/docs)
* [Streamlit](https://docs.streamlit.io/)