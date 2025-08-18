import matplotlib.pyplot as plt
from wordcloud import WordCloud
import os

def generate_graph(X, y, movie_name, path):
    # The method figure creates a new figure for plotting.
    # (10, 5): is the parameters, ta indicates the size.
    plt.figure(figsize=(10,5))
    # The barh function creates a horizontal bar chart.
    plt.barh(X, y)
    plt.xlabel(f'Similarity')
    plt.title(f'Top 5 similar movies to {movie_name}')
    # The gca function returns the current Axes instance on the current figure.
    # Invert_yaxis reverses the y-axis so that the highest values are at the top.
    plt.gca().invert_yaxis()
    plt.savefig(os.path.join(path, f'{movie_name}_similarity.png'))
    plt.show()

# Generating word cloud
def generate_wc(index, df, movie_name, path):
    # corpus is a list of text data for the word cloud
    # df.iloc[i]['corpus'] retrieves the 'corpus' column for the i-th row.
    corpus = [df.iloc[i]['corpus'] for i in index]
    # join the text from the corpus
    text = " ".join(corpus)
    # Generate the word cloud it is a class.
    # .generate() creates the word cloud from the text.
    wc = WordCloud(width=800, height=500, background_color='white').generate(text)
    plt.figure(figsize=(10, 5))
    # imshow displays the image, interpolation='bilinear' smoothens the image.
    plt.imshow(wc, interpolation='bilinear')
    # axis("off") turns off the axis.
    plt.axis("off")
    plt.savefig(os.path.join(path, f'{movie_name}_wordcloud.png'))
    plt.show()