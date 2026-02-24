from langchain_groq import ChatGroq
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate
from scripts.recommender import MovieRecommender
from langchain.agents import create_agent
from scripts.config import settings
import pandas as pd

recommender = MovieRecommender(use_embeddings=settings.USE_EMBEDDINGS)
movies_df = pd.read_csv(settings.DATA_PROCESSED / "movies_clean.csv")

@tool
def search_movies_by_description(query: str) -> str:
    """Busca filmes por descrição em linguagem natural, tema ou atmosfera."""
    results = recommender.recommend_by_query(query, top_n=5)
    if not results:
        return "Nenhum filme encontrado para essa descrição."
    return "\n".join([f"- {r['title'].title()} (similaridade: {r['score']:.2f})" for r in results])

@tool
def get_similar_movies(movie_title: str) -> str:
    """Retorna filmes similares a um título específico."""
    row = movies_df[movies_df['title'].str.lower() == movie_title.lower()]
    if row.empty:
        return f"Filme '{movie_title}' não encontrado na base de dados."
    movie_id = row.iloc[0]['movie_id']
    results = recommender.recommend_by_movie_id(movie_id, top_n=5)
    if not results:
        return "Nenhuma recomendação encontrada."
    return "\n".join([f"- {r['title'].title()}: {r['explanation']}" for r in results])

def build_agent():
    llm = ChatGroq(
        api_key=settings.GROQ_API_KEY,
        model=settings.GROQ_MODEL,
        temperature=0.3
    )
    tools = [search_movies_by_description, get_similar_movies]
    system_prompt = (
        "Você é um assistente especialista em recomendação de filmes. "
        "Use as ferramentas disponíveis para encontrar e recomendar filmes com base no pedido do usuário. "
        "Explique brevemente o motivo de cada recomendação. Responda sempre em Português."
    )
    return create_agent(llm, tools, system_prompt=system_prompt)