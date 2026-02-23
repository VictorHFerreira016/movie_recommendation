import numpy as np
import pandas as pd
from scripts.recommender import MovieRecommender
from scripts.config import settings

def precision_at_k(recommended, relevant, k):
    if k == 0 or not recommended:
        return 0.0
    recommended_k = recommended[:k]
    relevant_set = set(relevant)
    hits = sum(1 for r in recommended_k if r in relevant_set)
    return hits / k

def recall_at_k(recommended, relevant, k):
    if len(relevant) == 0:
        return 0.0
    recommended_k = recommended[:k]
    relevant_set = set(relevant)
    hits = sum(1 for r in recommended_k if r in relevant_set)
    return hits / len(relevant_set)

def average_precision(recommended, relevant):
    if not relevant:
        return 0.0
    relevant_set = set(relevant)
    hits = 0
    sum_prec = 0.0
    for i, rec in enumerate(recommended):
        if rec in relevant_set:
            hits += 1
            sum_prec += hits / (i + 1)
    return sum_prec / len(relevant_set)

def evaluate_recommender(recommender, test_queries, ground_truth, k=10):
    """
    test_queries: lista de IDs de filmes (int) ou textos (str) para consulta.
    ground_truth: dicionário {consulta: lista_de_ids_relevantes}
    """
    precisions, recalls, maps = [], [], []
    
    for query, relevant in ground_truth.items():
        if isinstance(query, int):
            recs, _ = recommender.recommend_by_movie_id(query, top_n=k)
        else:
            recs, _ = recommender.recommend_by_query(query, top_n=k)
        
        precisions.append(precision_at_k(recs, relevant, k))
        recalls.append(recall_at_k(recs, relevant, k))
        maps.append(average_precision(recs, relevant))
    
    return {
        f'Precision@{k}': np.mean(precisions),
        f'Recall@{k}': np.mean(recalls),
        f'MAP@{k}': np.mean(maps)
    }

def compare_methods(test_queries, ground_truth, k=10):
    print("Avaliando com TF-IDF...")
    rec_tfidf = MovieRecommender(use_embeddings=False)
    metrics_tfidf = evaluate_recommender(rec_tfidf, test_queries, ground_truth, k)
    
    print("Avaliando com Embeddings + FAISS...")
    rec_emb = MovieRecommender(use_embeddings=True)
    metrics_emb = evaluate_recommender(rec_emb, test_queries, ground_truth, k)
    
    df = pd.DataFrame([metrics_tfidf, metrics_emb], index=['TF-IDF', 'Embeddings'])
    print("\n=== Comparação de Métodos ===\n")
    print(df)
    
    df.to_csv(settings.DATA_PROCESSED / 'comparison_metrics.csv')
    return df

if __name__ == "__main__":
    movies_df = pd.read_csv(settings.DATA_PROCESSED / "movies_clean.csv")
    test_queries = [11, 12, 13] 
    ground_truth = {
        11: [22, 33, 44],
        12: [23, 34, 45],
        13: [24, 35, 46]
    }
    
    compare_methods(test_queries, ground_truth, k=10)