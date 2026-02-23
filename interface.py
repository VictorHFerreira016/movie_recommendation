import streamlit as st
import pandas as pd
from rapidfuzz import process, fuzz
from scripts.recommender import MovieRecommender
from scripts.config import settings
from scripts.visualization import generate_similarity_chart, generate_wordcloud_chart

st.set_page_config(page_title="Recomendação de Filmes", layout="wide", page_icon="🎬")

@st.cache_resource
def load_recommender():
    return MovieRecommender(use_embeddings=settings.USE_EMBEDDINGS)

@st.cache_data
def load_movies():
    df = pd.read_csv(settings.DATA_PROCESSED / "movies_clean.csv")
    if 'movie_id' in df.columns and 'id' not in df.columns:
        df = df.rename(columns={'movie_id': 'id'})
    return df['title'].str.title().tolist(), df

def fuzzy_search(query, choices, limit=10):
    if not query:
        return []
    results = process.extract(query, choices, scorer=fuzz.WRatio, limit=limit)
    return [match[0] for match in results if match[1] > 50]

def main():
    st.title("🎬 Sistema de Recomendação de Filmes")
    st.markdown("### Busca inteligente com explicações geradas por IA")
    
    movie_titles, movies_df = load_movies()
    recommender = load_recommender()
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        search_query = st.text_input(
            "🔍 Digite o nome do filme:", 
            placeholder="Ex: Dark Knight, Inception, Avatar..."
        )
    
    with col2:
        top_n = st.slider("Qtd. recomendações:", 3, 15, 10)
    
    if search_query:
        matches = fuzzy_search(search_query, movie_titles, limit=15)
        
        if matches:
            selected_movie = st.selectbox("📽️ Filmes encontrados:", matches, key="search_select")
        else:
            st.warning("⚠️ Nenhum filme encontrado. Tente outro termo.")
            return
    else:
        selected_movie = st.selectbox("📽️ Ou escolha da lista:", [""] + movie_titles, key="list_select")
    
    if not selected_movie:
        st.info("👆 Digite ou selecione um filme para começar")
        return
    
    if st.button("✨ Gerar Recomendações", type="primary", use_container_width=True):
        with st.spinner("🤖 Analisando e gerando explicações com IA..."):
            movie_row = movies_df[movies_df['title'].str.title() == selected_movie]
            
            if movie_row.empty:
                st.error("❌ Filme não encontrado no banco de dados")
                return
            
            movie_id = movie_row.iloc[0]['id']
            recommendations = recommender.recommend_by_movie_id(movie_id, top_n=top_n)
            
            if not recommendations:
                st.warning("⚠️ Nenhuma recomendação encontrada")
                return
            
            st.success(f"🎯 **{top_n}** recomendações para: **{selected_movie}**")
            st.divider()
            
            tab1, tab2, tab3 = st.tabs(["📋 Recomendações", "📊 Gráfico de Similaridade", "☁️ Nuvem de Palavras"])
            
            with tab1:
                st.markdown("#### 🎥 Filmes Recomendados")
                st.markdown("*Clique em 'Por que este filme?' para ver a análise da IA*")
                st.write("")
                
                for i, rec in enumerate(recommendations, 1):
                    col1, col2 = st.columns([4, 1])
                    
                    with col1:
                        st.markdown(f"**{i}. {rec['title'].title()}**")
                    
                    with col2:
                        st.metric("Similaridade", f"{rec['score']:.1%}")
                    
                    with st.expander(f"🤖 Por que **{rec['title'].title()}**?"):
                        st.markdown("**Análise da IA:**")
                        st.info(rec['explanation'])
                        
                        movie_info = movies_df[movies_df['id'] == rec['id']]
                        if not movie_info.empty:
                            genres = movie_info.iloc[0].get('genres', 'N/A')
                            overview = movie_info.iloc[0].get('overview', 'N/A')
                            
                            st.markdown("**Detalhes:**")
                            st.markdown(f"- **Gêneros:** {genres}")
                            if overview and overview != 'N/A':
                                st.markdown(f"- **Sinopse:** {overview[:200]}...")
                    
                    st.divider()
            
            with tab2:
                st.markdown("#### 📊 Comparação de Similaridade")
                titles = [rec['title'].title() for rec in recommendations]
                scores = [rec['score'] for rec in recommendations]
                
                chart_b64 = generate_similarity_chart(titles, scores, selected_movie)
                st.markdown(f'<img src="data:image/png;base64,{chart_b64}" style="width:100%"/>', 
                           unsafe_allow_html=True)
                
                st.caption("Quanto maior a barra, mais similar o filme é ao escolhido")
            
            with tab3:
                st.markdown("#### ☁️ Palavras-chave dos Filmes Recomendados")
                rec_ids = [rec['id'] for rec in recommendations]
                corpus_texts = movies_df[movies_df['id'].isin(rec_ids)]['corpus'].fillna('').tolist()
                
                if corpus_texts:
                    wc_b64 = generate_wordcloud_chart(corpus_texts)
                    st.markdown(f'<img src="data:image/png;base64,{wc_b64}" style="width:100%"/>', 
                               unsafe_allow_html=True)
                    st.caption("📌 Palavras maiores = termos mais frequentes nos filmes recomendados")
                else:
                    st.warning("⚠️ Dados insuficientes para gerar nuvem de palavras")
            
            with st.sidebar:
                st.markdown("### 📊 Estatísticas")
                st.metric("Total de recomendações", len(recommendations))
                avg_score = sum(r['score'] for r in recommendations) / len(recommendations)
                st.metric("Similaridade média", f"{avg_score:.1%}")
                
                st.markdown("---")
                st.markdown("### ℹ️ Sobre")
                method = "🧠 Embeddings + Pinecone" if settings.USE_EMBEDDINGS else "📝 TF-IDF"
                st.markdown(f"**Método:** {method}")
                st.markdown(f"**Modelo LLM:** {settings.GROQ_MODEL}")

if __name__ == "__main__":
    main()