from enum import Enum


class SentimentModel(Enum):
    positive: str = "positive"
    negative: str = "negative"
    neutral: str = "neutral"
