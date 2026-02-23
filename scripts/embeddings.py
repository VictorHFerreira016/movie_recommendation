import numpy as np
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone, ServerlessSpec
import logging
from scripts.config import settings

logging.basicConfig(level=logging.INFO)

def generate_embeddings(texts, batch_size=32):
    model = SentenceTransformer(settings.EMBEDDING_MODEL)
    embeddings = model.encode(texts, batch_size=batch_size, show_progress_bar=True)
    return embeddings

def init_pinecone():
    pc = Pinecone(api_key=settings.PINECONE_API_KEY)
    
    if settings.PINECONE_INDEX_NAME not in pc.list_indexes().names():
        pc.create_index(
            name=settings.PINECONE_INDEX_NAME,
            dimension=settings.EMBEDDING_DIMENSION,
            metric='cosine',
            spec=ServerlessSpec(cloud='aws', region=settings.PINECONE_ENVIRONMENT)
        )
        logging.info(f"Índice {settings.PINECONE_INDEX_NAME} criado")
    
    return pc.Index(settings.PINECONE_INDEX_NAME)

def save_to_pinecone(movies_df, text_column='corpus'):
    texts = movies_df[text_column].fillna('').tolist()
    movie_ids = movies_df['movie_id'].tolist()
    
    logging.info("Gerando embeddings...")
    embeddings = generate_embeddings(texts)
    
    logging.info("Salvando no Pinecone...")
    index = init_pinecone()
    
    vectors = []
    for i, (movie_id, emb) in enumerate(zip(movie_ids, embeddings)):
        metadata = {
            'title': movies_df.iloc[i]['title'],
            'corpus': movies_df.iloc[i]['corpus'][:1000],
            'genres': movies_df.iloc[i].get('genres', ''),
            'overview': movies_df.iloc[i].get('overview', '')[:500]
        }
        vectors.append((str(movie_id), emb.tolist(), metadata))
        
        if len(vectors) >= 100:
            index.upsert(vectors=vectors, namespace=settings.PINECONE_NAMESPACE)
            vectors = []
    
    if vectors:
        index.upsert(vectors=vectors, namespace=settings.PINECONE_NAMESPACE)
    
    logging.info(f"Dados salvos no Pinecone")
    
    settings.EMBEDDINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
    
    np.save(settings.EMBEDDINGS_PATH, embeddings)
    np.save(settings.MOVIE_IDS_PATH, movie_ids)

def get_pinecone_index():
    pc = Pinecone(api_key=settings.PINECONE_API_KEY)
    return pc.Index(settings.PINECONE_INDEX_NAME)