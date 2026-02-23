import matplotlib.pyplot as plt
from wordcloud import WordCloud
import io
import base64

def generate_similarity_chart(titles, scores, movie_name):
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(titles, scores, color='#1f77b4')
    
    for bar in bars:
        width = bar.get_width()
        ax.text(width, bar.get_y() + bar.get_height()/2, 
                f'{width:.1%}', 
                ha='left', va='center', fontsize=9, color='black')
    
    ax.set_xlabel('Similaridade', fontsize=12)
    ax.set_title(f'Top {len(titles)} filmes similares a "{movie_name}"', fontsize=14, fontweight='bold')
    ax.invert_yaxis()
    ax.set_xlim(0, max(scores) * 1.15)
    plt.tight_layout()
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    buf.seek(0)
    plt.close(fig)
    
    return base64.b64encode(buf.read()).decode()

def generate_wordcloud_chart(corpus_texts):
    text = " ".join(corpus_texts)
    
    wc = WordCloud(
        width=800, 
        height=400, 
        background_color='white',
        colormap='viridis',
        max_words=50,
        min_font_size=10
    ).generate(text)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.imshow(wc, interpolation='bilinear')
    ax.axis("off")
    ax.set_title("Palavras-chave dos filmes recomendados", fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    buf.seek(0)
    plt.close(fig)
    
    return base64.b64encode(buf.read()).decode()