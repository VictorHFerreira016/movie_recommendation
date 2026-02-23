import pandas as pd
import json
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import logging
from scripts.config import settings

def generate_embeddings_if_needed(df_merged):
    if settings.USE_EMBEDDINGS:
        logging.info("USE_EMBEDDINGS=True: enviando para Pinecone...")
        from scripts.embeddings import save_to_pinecone
        save_to_pinecone(df_merged, text_column='corpus')
    else:
        logging.info("USE_EMBEDDINGS=False: pulando embeddings.")

"""This function allows configure the log system once, the parameter level, define the minimun level 
that a log message needs to be shown. It confirms that things are working correctly. 
The parameter format, controls how the log messages are gonna be shown.

%s: the "%" indicates where a value will be inserted, and the data type is specified after the "%". "s" 
indicates string.
%(levelname)s: This is a variable that will be replaced with the name of the message's severity level.
%(message)s: This is the variable that will be replaced with the actual log message you passed to the 
function."""

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logging.info("O programa iniciou.")

"""Function to clean and prepare data to the cossine similarity function.
args:
    df_credits: dataset to be cleaned."""
def clean_df_credits(df_credits):
    logging.info("Starting to clean the credits DataFrame.")
    try: 
        df_credits.drop(columns='id', inplace=True)
    except KeyError:
        logging.warning("The column 'movie_id' was not found to be removed. Continuing...")
    
    def parse_crew_json(crew):
        try:
            return json.loads(crew)
        except (json.JSONDecodeError, TypeError):
            logging.error(f"Unable to decode JSON for crew: {crew}")
            return []
        
    df_credits['crew'] = df_credits['crew'].apply(parse_crew_json)
    
    def get_director(crew_list):
        for i in crew_list:
            try:
                if i.get('job', '').lower() == 'director':
                    return i['name']
            except (TypeError, AttributeError):
                logging.warning(f"Unspected formate: {i}")
                continue

    df_credits['director'] = df_credits['crew'].apply(get_director)

    def get_screenwriter(crew_list):
        screenwriters = {'screenwriter', 'screenplay', 'writer', 'author', 'co-writer', 'story', 'adaptation'}
        for i in crew_list:
            try:
                if i.get('job', '').lower() in screenwriters:
                    return i['name']
            except (TypeError, AttributeError):
                logging.warning(f"Unspected format: {i}")
                continue 
        return None

    df_credits['screenwriter'] = df_credits['crew'].apply(get_screenwriter)

    def get_top_actors(cast_json, n=3):
        try: 
            cast = json.loads(cast_json)
            top_cast = sorted(cast, key=lambda x: x.get('order', 999))[:n]
            return " ".join([i.get('name', '') for i in top_cast])

        except (json.JSONDecodeError, TypeError):
            logging.error(f"It was not possible to convert JSON: {cast}")

    df_credits['cast'] = df_credits['cast'].apply(lambda x: get_top_actors(x, n=3))

    df_credits = df_credits.drop(columns='crew')
    df_credits['director'] = df_credits['director'].fillna("Unknown Director")
    df_credits['screenwriter'] = df_credits['screenwriter'].fillna("Unknown Director")

    return df_credits

def clean_df_movies(df_movies):
    """Function to clean and prepare the movie DataFrame"""
    logging.info("Starting cleaning...")

    cols_to_drop = ['budget', 'homepage', 'id', 'revenue', 'status', 'vote_count', 'vote_average', 
                    'runtime', 'release_date', 'popularity', 'keywords']
    verified_cols = [col for col in cols_to_drop if col in df_movies.columns]
    df_movies = df_movies.drop(columns=verified_cols)
    logging.info(f"Removed columns: {verified_cols}")

    def extract_names(data, column):
        try:
            item = json.loads(data)
            if not isinstance(item, list):
                logging.warning(f"Data from column {column} is not a list.")
                return ""
            return ', '.join([i.get('name', '') for i in item])
        except (json.JSONDecodeError, TypeError):
            logging.error(f"Error to convert JSON on column {column}: {data}")
            return ''
    
    json_columns = ['genres', 'production_companies', 'production_countries', 'spoken_languages']
    for i in json_columns:
        if i in df_movies.columns:
            df_movies[i] = df_movies[i].apply(lambda x: extract_names(x, i))

    def to_lower(x):
        if pd.isna(x):
            return ""
        if isinstance(x, str):
            return x.lower()
        if isinstance(x, list):
            return [to_lower(i) for i in x]
        else:
            return str(x).lower()
        
    for col in df_movies.columns:
        if col != 'corpus':
            df_movies[col] = df_movies[col].apply(to_lower)

    df_movies['overview'] = df_movies['overview'].fillna('')

    first_col = json_columns[0]
    if first_col in df_movies.columns:
        df_movies['corpus'] = df_movies[first_col].astype(str)
    else:
        df_movies['corpus'] = ''
    
    for i in json_columns[1:]:
        if i in df_movies.columns:
            df_movies['corpus'] = df_movies['corpus'] + ' ' + df_movies[i].astype(str)
        else:
            logging.warning(f"The column {i} to create the corpus was not found")

    logging.info("The DataFrame movie was cleaned!")

    return df_movies

from scripts.visualization import generate_graph, generate_wc

def recommend_movie_with_scores(movie_index, cosine_sim, df, top_n):
    try:
        movie_name = df.iloc[movie_index]['title']
        logging.info(f"Generating recomendations for the movie: {movie_name}")
        sim_scores = list(enumerate(cosine_sim[movie_index]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        sim_scores = sim_scores[1:top_n+1]

        index = [i[0] for i in sim_scores]
        scores = [i[1] for i in sim_scores]
        titles = df.iloc[index]['title'].tolist()

        return titles, scores
    
    except IndexError:
        logging.error(f"Error: the index {movie_index} is out of the range.")
        return None, None
    except Exception as e:
        logging.critical(f"An unspected error occured: {e}")
        return None, None
    
"""Function to calculate cossine similarity."""
def calculate(df_movies):
    tfVectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 2), max_features=20000)
    tfidf_matrix = tfVectorizer.fit_transform(df_movies['corpus'])
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    logging.info("Cosine similarity calculated (shape=%s)", cosine_sim.shape)
    return cosine_sim