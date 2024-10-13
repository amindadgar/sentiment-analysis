import logging
import json

from analyzer.sentiment import AnalyzeSentiment
from schema import SentimentModel
from utils.processor import Processor

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    processor = Processor()
    analyzer = AnalyzeSentiment(load_openai=True)

    # the `result.json` is the telegram exported data
    data = processor.process_telegram_json("result.json")

    user_data: dict[str, dict[SentimentModel, float]] = {}

    for idx, (user, raw_messages) in enumerate(data.items()):
        logging.info(
            f"Processing user {user} messages. idx: {idx + 1}/{len(data.keys())}"
        )

        user_sentiments = analyzer.process(texts=raw_messages)
        user_data[user] = user_sentiments

    # saving the sentiments
    user_data_serializable = {
        user: {sentiment.name: score for sentiment, score in scores.items()}
        for user, scores in user_data.items()
    }
    with open("user_data_llm.json", "w") as file:
        json.dump(user_data_serializable, file, indent=4)
