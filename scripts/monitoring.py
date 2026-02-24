import json
import logging
from datetime import datetime
from pathlib import Path
from scripts.config import settings

LOG_PATH = Path("logs/recommendation_log.jsonl")
LOG_PATH.parent.mkdir(exist_ok=True)

def log_recommendation_event(
        movie_id: int, 
        method: str, 
        top_n: int, 
        scores: list[float]
    ):
    avg_score = sum(scores) / len(scores) if