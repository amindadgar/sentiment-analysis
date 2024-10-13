import json
import numpy as np

from analyzer.sentiment import AnalyzeSentiment
from schema import SentimentModel
from utils.processor import Processor

if __name__ == "__main__":
    processor = Processor()
    analyzer = AnalyzeSentiment()

    # the `result.json` is the telegram exported data
    data = processor.process_telegram_json("result.json")

    user_data: dict[str, dict[SentimentModel, float]] = {}

    for idx, (user, raw_messages) in enumerate(data.items()):
        print(f"Processing user {user} messages. idx: {idx + 1}/{len(data.keys())}")

        user_sentiments = analyzer.process_hezarai(texts=raw_messages)
        user_data[user] = user_sentiments

    # saving the sentiments
    user_data
    with open("user_data.json", "w") as file:
        json.dumps(user_data, file, indent=4)

    # separating the sentiments for any future use
    # users = list(user_data.keys())
    # positive_scores = [user_data[user].get(SentimentModel.positive, np.nan) for user in users]
    # negative_scores = [user_data[user].get(SentimentModel.negative, np.nan) for user in users]
    # neutral_scores = [user_data[user].get(SentimentModel.neutral, np.nan) for user in users]
