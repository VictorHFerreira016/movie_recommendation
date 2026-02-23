import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
import logging
from scripts.config import settings
from scripts.embeddings import get_pinecone_index

logging.basicConfig(level=logging.INFO)

class MovieRecommender:
    def __init__(self, use_embeddings=None):
        self.use_embeddings = settings.USE_EMBEDDINGS if use_embeddings is None else use_embeddings
        self.movies_df = pd.read_csv(settings.DATA_PROCESSED / "movies_clean.csv")
        if 'movie_id' in self.movies_df.columns and 'id' not in self.movies_df.columns:
            self.movies_df.rename(columns={'movie_id': 'id'}, inplace=True)
        
        self.movie_ids = np.load(settings.MOVIE_IDS_PATH, allow_pickle=True).tolist()
        
        if len(self.movie_ids) != len(self.movies_df):
            raise ValueError("Mismatch between number of movie IDs and DataFrame rows")
        self.movies_df['id'] = self.movie_ids
        
        if self.use_embeddings:
            logging.info("Conectando ao Pinecone...")
            self.index = get_pinecone_index()
            self.model = SentenceTransformer(settings.EMBEDDING_MODEL)
            
            if settings.ENABLE_EXPLANATIONS:
                self.llm = ChatGroq(
                    api_key=settings.GROQ_API_KEY, # type: ignore
                    model=settings.GROQ_MODEL,
                    temperature=settings.GROQ_TEMPERATURE
                )
        else:
            logging.info("Usando TF-IDF...")
            self.tfidf_vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 2), max_features=20000)
            self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(self.movies_df['corpus'])
            self.movies_df = self.movies_df.reset_index(drop=True)
        
    def _generate_explanation(self, original_movie, recommended_movie, score, original_genres="", rec_genres="", original_overview="", rec_overview=""):
        if not settings.ENABLE_EXPLANATIONS:
            return f"Similaridade: {score:.2%}"
        
        try:
            prompt = PromptTemplate(
                input_variables=["movie1", "movie2", "score", "genres1", "genres2", "overview1", "overview2"],
                template="""Você é um especialista em cinema. Explique em 2-3 frases curtas e objetivas por que '{movie2}' é similar a '{movie1}' (similaridade: {score:.0%}).

    Filme 1: {movie1}
    Gêneros: {genres1}
    Sinopse: {overview1}

    Filme 2: {movie2}
    Gêneros: {genres2}
    Sinopse: {overview2}

    Foque em: gêneros compartilhados, temas narrativos, tom/atmosfera, ou elementos visuais/estilísticos comuns. Seja específico e natural.

    Explicação:"""
            )
            
            chain = prompt | self.llm
            response = chain.invoke({
                "movie1": original_movie,
                "movie2": recommended_movie,
                "score": score,
                "genres1": original_genres or "Não especificado",
                "genres2": rec_genres or "Não especificado",
                "overview1": original_overview[:300] or "Não disponível",
                "overview2": rec_overview[:300] or "Não disponível"
            })
            
            return response.content.strip()
        except Exception as e:
            logging.error(f"Erro ao gerar explicação: {e}")
            return f"Similaridade: {score:.2%}"
        
    def recommend_by_movie_id(self, movie_id, top_n=10):
        if self.use_embeddings:
            query_results = self.index.query(
                id=str(movie_id),
                top_k=top_n + 1,
                namespace=settings.PINECONE_NAMESPACE,
                include_metadata=True
            )
            
            original_row = self.movies_df[self.movies_df['id'] == movie_id].iloc[0]
            original_title = original_row['title']
            original_genres = original_row.get('genres', '')
            original_overview = original_row.get('overview', '')
            
            recommendations = []
            for match in query_results['matches'][1:top_n+1]:
                rec_id = int(match['id'])
                score = match['score']
                rec_title = match['metadata']['title']
                rec_genres = match['metadata'].get('genres', '')
                rec_overview = match['metadata'].get('overview', '')
                
                explanation = self._generate_explanation(
                    original_title, rec_title, score,
                    original_genres, rec_genres,
                    original_overview, rec_overview
                )
                recommendations.append({
                    'id': rec_id,
                    'title': rec_title,
                    'score': score,
                    'explanation': explanation
                })
            
            return recommendations
        else:
            idx = self.movies_df[self.movies_df['id'] == movie_id].index
            if len(idx) == 0:
                return []
            idx = idx[0]
            
            sim_scores = list(enumerate(cosine_similarity(self.tfidf_matrix[idx], self.tfidf_matrix)[0]))
            sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:top_n+1]
            
            original_row = self.movies_df.iloc[idx]
            original_title = original_row['title']
            original_genres = original_row.get('genres', '')
            original_overview = original_row.get('overview', '')
            
            recommendations = []
            for movie_idx, score in sim_scores:
                rec_row = self.movies_df.iloc[movie_idx]
                rec_id = rec_row['id']
                rec_title = rec_row['title']
                rec_genres = rec_row.get('genres', '')
                rec_overview = rec_row.get('overview', '')
                
                explanation = self._generate_explanation(
                    original_title, rec_title, score,
                    original_genres, rec_genres,
                    original_overview, rec_overview
                )
                
                recommendations.append({
                    'id': rec_id,
                    'title': rec_title,
                    'score': score,
                    'explanation': explanation
                })
            
            return recommendations
        
    def recommend_by_query(self, query_text, top_n=10):
        if self.use_embeddings:
            query_emb = self.model.encode([query_text])
            
            results = self.index.query(
                vector=query_emb[0].tolist(),
                top_k=top_n,
                namespace=settings.PINECONE_NAMESPACE,
                include_metadata=True
            )
            
            recommendations = []
            for match in results['matches']:
                recommendations.append({
                    'id': int(match['id']),
                    'title': match['metadata']['title'],
                    'score': match['score'],
                    'explanation': f"Correspondência: {match['score']:.2%}"
                })
            
            return recommendations
        else:
            query_vec = self.tfidf_vectorizer.transform([query_text])
            sim_scores = cosine_similarity(query_vec, self.tfidf_matrix).flatten()
            top_indices = sim_scores.argsort()[-top_n:][::-1]
            
            recommendations = []
            for idx in top_indices:
                recommendations.append({
                    'id': self.movies_df.iloc[idx]['id'],
                    'title': self.movies_df.iloc[idx]['title'],
                    'score': sim_scores[idx],
                    'explanation': f"Correspondência: {sim_scores[idx]:.2%}"
                })
            
            return recommendations