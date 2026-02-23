import pandas as pd
import logging
from scripts.preprocessing import clean_df_credits, clean_df_movies, generate_embeddings_if_needed
from scripts.config import settings
from scripts.recommender import MovieRecommender
import argparse

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

def pipeline(method=None):
    """
    method: 'tfidf' or 'embeddings'. If None, it uses the value of config.USE_EMBEDDINGS.
    """
    logging.info("Starting pipeline.")
    
    try:
        logging.info("Carregando datasets...")

        df_credits = pd.read_csv(settings.DATA_RAW / "tmdb_5000_credits.csv")
        df_movies = pd.read_csv(settings.DATA_RAW / "tmdb_5000_movies.csv")

    except FileNotFoundError as e:

        logging.error(f"Erro ao carregar dados: {e}")
        return

    logging.info("Limpando dados...")

    df_credits_cleaned = clean_df_credits(df_credits)
    df_movies_cleaned = clean_df_movies(df_movies)
    df_credits_cleaned['title'] = df_credits_cleaned['title'].str.lower()
    df_movies_cleaned['title'] = df_movies_cleaned['title'].str.lower()
    df_merged = pd.merge(df_movies_cleaned, df_credits_cleaned, on='title', how='inner')

    logging.info(f"Merge concluído! Shape: {df_merged.shape}")

    df_merged['id'] = df_merged['movie_id']

    df_merged['corpus'] = (
        df_merged['corpus'].astype(str) + ' ' +
        df_merged['cast'].astype(str) + ' ' +
        df_merged['director'].astype(str) + ' ' +
        df_merged['screenwriter'].astype(str)
    )

    df_merged.to_csv(settings.DATA_PROCESSED / "movies_clean.csv", index=False)

    logging.info("DataFrame limpo salvo em data/processed/movies_clean.csv")

    generate_embeddings_if_needed(df_merged)

    if method == 'tfidf':
        use_emb = False
    elif method == 'embeddings':
        use_emb = True
    else:
        use_emb = settings.USE_EMBEDDINGS 

    recommender = MovieRecommender(use_embeddings=use_emb)

    movie_title = 'The Dark Knight Rises'
    movie_row = df_merged[df_merged['title'] == movie_title.lower()]
    if movie_row.empty:

        logging.error(f"Filme '{movie_title}' não encontrado.")
        return

    movie_id = movie_row.iloc[0]['movie_id']  

    logging.info(f"Gerando recomendações para '{movie_title}' (ID: {movie_id}) usando método: {'embeddings' if use_emb else 'tfidf'}")
    
    recommendations = recommender.recommend_by_movie_id(movie_id, top_n=10)

    if recommendations:
        print("\n--- Recomendações ---")
        for i, rec in enumerate(recommendations, 1):
            print(f"{i}. {rec['title'].title()}")
            print(f"   Similaridade: {rec['score']:.2%}")
            print(f"   {rec['explanation']}")
            print()
        print("---------------------\n")
    else:
        logging.warning("Nenhuma recomendação encontrada.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Recomendação de filmes")
    parser.add_argument('--method', choices=['tfidf', 'embeddings'], help='Método de recomendação (tfidf ou embeddings)')
    args = parser.parse_args()
    pipeline(method=args.method)