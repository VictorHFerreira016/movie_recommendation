import gradio as gr
import pandas as pd
from pathlib import Path
from scripts.utils import save_artifact, load_artifact

from scripts.config import DATA_RAW
from scripts.preprocessing import clean_df_credits, clean_df_movies, calculate

# ARTIFACTS_DIR is the directory where processed artifacts are stored. It gets from the function Path, so it is created if it doesn't exist.
ARTIFACTS_DIR = Path("artifacts")
# exist_ok=True allows the directory to be created if it doesn't exist.
ARTIFACTS_DIR.mkdir(exist_ok=True)

# ARTIFACTS_DIR is an object, so we can use the / operator to create paths.
DF_PATH = ARTIFACTS_DIR / "df_processed.joblib"
SIM_PATH = ARTIFACTS_DIR / "cosine_sim.joblib"

# from config, DATA_RAW retrieves the raw data paths.
CREDITS_CSV_PATH = DATA_RAW / "tmdb_5000_credits.csv"
MOVIES_CSV_PATH = DATA_RAW / "tmdb_5000_movies.csv"

# Load and prepare the data for the interface.
def load_and_prepare():
    # verifies if processed artifacts exist, if not it loads and processes the raw data.
    if DF_PATH.exists() and SIM_PATH.exists():
        df = load_artifact(DF_PATH)
        cosine_sim = load_artifact(SIM_PATH)
    else:
        df_credits = pd.read_csv(CREDITS_CSV_PATH)
        df_movies = pd.read_csv(MOVIES_CSV_PATH)
        df_credits_clean = clean_df_credits(df_credits)
        df_movies_clean = clean_df_movies(df_movies)
        # Convert 'title' columns to lowercase for case-insensitive matching
        # The method get() retrieves a value from a DataFrame column, returning a default if the column is not found.
        # '' means an empty string will be used if the column is not found.
        df_credits_clean['title'] = df_credits_clean.get('title', '').astype(str).str.lower()
        df_movies_clean['title'] = df_movies_clean.get('title', '').astype(str).str.lower()
        # Merge the two DataFrames on the 'title' column, using an inner join to keep only matching titles.
        # 'inner' means that only rows with matching titles in both DataFrames will be kept.
        df = pd.merge(df_movies_clean, df_credits_clean, on='title', how='inner')
        # Creating corpus.
        # strip() removes leading and trailing whitespace.
        df['corpus'] = (
            df.get('corpus', '').astype(str) + ' ' +
            df.get('cast', '').astype(str) + ' ' +
            df.get('director', '').astype(str) + ' ' +
            df.get('screenwriter', '').astype(str)
        ).str.strip()
        cosine_sim = calculate(df)
        save_artifact(df, DF_PATH)
        save_artifact(cosine_sim, SIM_PATH)
    # reset_index() is used to reset the index of the DataFrame. 
    # It means that the old index is discarded and a new sequential index is created.
    # drop=True means that the old index is not added as a column in the new DataFrame.
    return df.reset_index(drop=True), cosine_sim

df, cosine_sim = load_and_prepare()

def get_recommendations(movie_title, top_n):
    # idxs is a list of indices where the movie title matches
    # index[] is used to access the index of the DataFrame
    # df['title'] is the column we are interested in, and df['title'] == movie_title.lower() 
    # checks for matches
    # tolist() converts the index to a list
    idxs = df.index[df['title'] == movie_title.lower()].tolist()
    # If no matches are found, return a message and None for the wordcloud
    if not idxs:
        return "Movie not found.", None
    # idx is the index of the matching movie
    idx = idxs[0]
    # enumerate() is used to get the index and value of the cosine similarity scores
    # cosine_sim[idx] retrieves the cosine similarity scores for the matching movie
    # so sims is a list of tuples (index, score)
    sims = list(enumerate(cosine_sim[idx]))
    # sorted() is used to sort the similarity scores in descending order
    # the first parameter sims is the list to be sorted
    # the second parameter key specifies a function of one argument that is used to 
    # extract a comparison key from each element in the list
    # so x[1] retrieves the score
    sims = sorted(sims, key=lambda x: x[1], reverse=True)
    # Get the top N recommendations
    sims = sims[1:top_n+1]
    # the list comprehension gets the indexes of the top_n 
    indices = [i for i, s in sims]
    titles = df.iloc[indices]['title'].str.title().tolist()
    # the list comprehension gets the scores of the top_n
    # :.4f is a format specifier that limits the number of decimal places to 4
    scores = [f"{s:.4f}" for i, s in sims]
    # zip() is used to combine the titles and scores into pairs,
    # it returns a list of tuples
    # and then we join them into a string with the format "title (score)"
    result = "\n".join([f"{t} (score: {s})" for t, s in zip(titles, scores)])
    try:
        from wordcloud import WordCloud
        text = " ".join(df.iloc[indices]['corpus'].astype(str).tolist())
        wc = WordCloud(width=800, height=400, background_color='white', max_words=150).generate(text)
        # it returns the result and the wordcloud, the use of "to_array()" is to convert the image to a format suitable for display
        # because wc at this moment is number
        return result, wc.to_array() 
    except Exception:
        return result, None

# Function to search for movies based on a query
def search_movies(title):
    title = title.lower()
    # first the list comprehension gets the unique titles, example: "the godfather"
    # and then verifies if the title is in the unique title
    return [t for t in df['title'].unique() if title in t]

# Blocks() is a Gradio component that allows you to create a user interface with multiple components.
with gr.Blocks() as demo:
    gr.Markdown("# Movie Recommendation")
    movie_select = gr.Dropdown(
        choices=sorted(df['title'].str.title().unique()),
        label="Choose a movie"
    )
    top_n = gr.Slider(1, 20, value=5, label="Number of recommendations")
    output_recs = gr.Textbox(label="Similar movies")
    output_cloud = gr.Image(label="Wordcloud")

    gr.Button("Recommend").click(
        # fn specifies the function to be called when the button is clicked
        fn=get_recommendations,
        inputs=[movie_select, top_n],
        outputs=[output_recs, output_cloud]
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0")
    # to run local
    # demo.launch()