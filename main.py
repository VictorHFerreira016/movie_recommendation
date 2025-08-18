import pandas as pd
import logging
from scripts.preprocessing import clean_df_credits, clean_df_movies, recommend_movie_with_scores, calculate
from scripts.config import DATA_RAW

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

def pipeline():
    logging.info("Starting pipeline.")
    try:
        logging.info("Carregando os datasets...")
        df_credits = pd.read_csv(DATA_RAW / "tmdb_5000_credits.csv")    
        df_movies = pd.read_csv(DATA_RAW / "tmdb_5000_movies.csv")
        logging.info("Datasets successfully loaded.")
    except FileNotFoundError as e:
        logging.error(f"Error to load data: {e}.")

    logging.info("Starting data processing...")
    df_credits_cleaned = clean_df_credits(df_credits)
    df_movies_cleaned = clean_df_movies(df_movies)
    logging.info("Load successfully!")

    df_credits_cleaned['title'] = df_credits_cleaned['title'].str.lower()
    df_movies_cleaned['title'] = df_movies_cleaned['title'].str.lower()
    df_merged = pd.merge(df_movies_cleaned, df_credits_cleaned, on='title', how='inner')
    logging.info(f"DataFrames merged! Shape: {df_merged.shape}")

    df_merged['corpus'] = (
        df_merged['corpus'].astype(str) + ' ' +
        df_merged['cast'].astype(str) + ' ' +
        df_merged['director'].astype(str) + ' ' +
        df_merged['screenwriter'].astype(str)
    )

    logging.info("Calculating cossine similarity...")
    cosine_sim_matrix = calculate(df_merged)
    logging.info("Finished!")

    try:
        movie_title = 'The Dark Knight Rises'
        movie_index = df_merged[df_merged['title'] == movie_title.lower()].index[0]
        logging.info(f"Gerating recommendations for '{movie_title}' (index: {movie_index}).")
        
        titles, scores = recommend_movie_with_scores(
            movie_index=movie_index,
            cosine_sim=cosine_sim_matrix,
            df=df_merged,
            top_n=10
        )

        if titles:
            print("\n--- Recommendations ---")
            for title, score in zip(titles, scores):
                print(f"- {title.title()} (Similarity: {score:.4f})")
            print("---------------------\n")

    except IndexError:
        logging.error(f"The movie '{movie_title}' was not found on the dataset.")
    
    logging.info("Pipeline successfully completed!")

if __name__ == "__main__":
    pipeline()