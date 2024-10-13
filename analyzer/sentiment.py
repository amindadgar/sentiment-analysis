import logging
from dotenv import load_dotenv

from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer, AutoConfig
import numpy as np
from scipy.special import softmax
from schema import SentimentModel
from hezar.models import Model
from openai import OpenAI
import re


class AnalyzeSentiment:
    def __init__(
        self,
        load_openai: bool = False,
        load_roberta: bool = False,
        load_hezarai: bool = False,
    ) -> None:
        # Ensure exactly one model is loaded
        models_to_load = [load_openai, load_roberta, load_hezarai]
        if sum(models_to_load) != 1:
            raise ValueError("Exactly one model should be loaded!")

        # Initialize based on the selected model
        self.tokenizer = None
        self.config = None
        self.roberta_model = None
        self.hezarai_model = None
        self.openai_client = None

        if load_roberta:
            model_name = "cardiffnlp/twitter-roberta-base-sentiment-latest"
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.config = AutoConfig.from_pretrained(model_name)
            self.roberta_model = AutoModelForSequenceClassification.from_pretrained(
                model_name
            )

        elif load_hezarai:
            self.hezarai_model = Model.load("hezarai/bert-fa-sentiment-dksf")

        elif load_openai:
            load_dotenv()
            self.openai_client = OpenAI()

    def process(self, texts: list[str]) -> dict[SentimentModel, float]:
        """
        process data using the enabled model

        Parametes
        ----------
        texts : list[str]
            a list of texts to analyze their sentiment

        Returns
        --------
        scores_dict : list[dict[SentimentModel, float]]
            a dictionary of sentiments `negative`, `positive`, or `neutral`
            each representative of average score between 0 and 1
        """
        scores_dict: list[dict[SentimentModel, float]]
        if self.roberta_model:
            scores_dict = self.process_using_roberta_model(texts=texts)
        if self.hezarai_model:
            scores_dict = self.process_hezarai(texts=texts)
        if self.openai_client:
            scores_dict = self.process_openai(texts=texts)

        return scores_dict

    def process_using_roberta_model(
        self, texts: list[str]
    ) -> dict[SentimentModel, float]:
        """
        Process a text for each sentiments using `cardiffnlp/twitter-roberta-base-sentiment-latest` model

        Parametes
        ----------
        texts : list[str]
            a list of texts to analyze their sentiment

        Returns
        --------
        scores_dict : list[dict[SentimentModel, float]]
            a dictionary of sentiments `negative`, `positive`, or `neutral`
            each representative of average score between 0 and 1
        """
        scores_dict: dict[SentimentModel, list[float]] = {
            SentimentModel.positive: [],
            SentimentModel.negative: [],
            SentimentModel.neutral: [],
        }

        for message in texts:
            message_sentiments = self._process_roberta(message)
            for sentiment, score in message_sentiments.items():
                scores_dict[sentiment].append(score)

        # averaging the score
        for sentiment in scores_dict:
            scores_dict[sentiment] = np.average(scores_dict[sentiment])

        return scores_dict

    def _process_roberta(self, text: str) -> dict[SentimentModel, float]:
        """
        Process a text for each sentiments using `cardiffnlp/twitter-roberta-base-sentiment-latest` model

        Parameters
        -----------
        text : str
            the text to be sentimented

        Returns
        --------
        scores_dict : dict[SentimentModel, float]
            a dictionary of sentiments `negative`, `positive`, or `neutral`
            each representative of their score between 0 and 1
        """
        text = self.preprocess(text)
        encoded_input = self.tokenizer(text, return_tensors="pt")
        output = self.roberta_model(**encoded_input)
        scores = output[0][0].detach().numpy()
        scores = softmax(scores)

        scores_dict = {}

        ranking = np.argsort(scores)
        ranking = ranking[::-1]
        for i in range(scores.shape[0]):
            label = self.config.id2label[ranking[i]]
            score = scores[ranking[i]]
            scores_dict[SentimentModel(label)] = np.round(float(score), 4)

        return scores_dict

    def process_hezarai(
        self, texts: list[str], batch_size: int = 100
    ) -> dict[SentimentModel, float]:
        """
        process using hezarai `hezarai/bert-fa-sentiment-dksf` sentiment model

        Parametes
        ----------
        texts : list[str]
            a list of texts to analyze their sentiment

        Returns
        --------
        scores_dict : list[dict[SentimentModel, float]]
            a dictionary of sentiments `negative`, `positive`, or `neutral`
            each representative of average score between 0 and 1
        """
        # Initialize score dictionary
        scores_dict: dict[SentimentModel, list[float]] = {
            SentimentModel.positive: [],
            SentimentModel.negative: [],
            SentimentModel.neutral: [],
        }

        # Process texts in batches
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            outputs = self.hezarai_model.predict(batch)

            # Process each output in the batch
            for output in outputs:
                label = output[0]["label"]
                score = output[0]["score"]

                scores_dict[SentimentModel(label)].append(np.round(float(score), 4))

        averaged_scores = self._average_scores(scores_dict=scores_dict)

        return averaged_scores

    def process_openai(self, texts: list[str]) -> dict[SentimentModel, float]:
        """
        process multiple texts using openai llm. return the average sentiment values

        TODO: Make async or do batch API
        """
        scores_dict: dict[SentimentModel, list[float]] = {
            SentimentModel.positive: [],
            SentimentModel.negative: [],
            SentimentModel.neutral: [],
        }

        for text in texts:
            try:
                text_sentiment = self._process_openai(text=text)

                for sentiment, score in text_sentiment.items():
                    scores_dict[sentiment].append(score)
            except Exception as exp:
                logging.error(f"Exception: {exp} during processing text: {text}!")

        averaged_scores_dict = self._average_scores(scores_dict=scores_dict)

        return averaged_scores_dict

    def _process_openai(self, text: str) -> dict[SentimentModel, float]:
        """
        process a given text using openai llm
        """
        completion = self.openai_client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are an expert assistant for sentiment analysis. "
                        "Your task is to analyze the sentiment of the provided text. "
                        "Your response should include two parts: "
                        "a single word representing the overall sentiment "
                        "('positive', 'negative', or 'neutral') and a sentiment score ranging from -1 to 1. "
                        "A score close to -1 indicates a strong negative sentiment, 0 indicates a neutral sentiment, "
                        "and a score close to 1 indicates a strong positive sentiment."
                    ),
                },
                {"role": "user", "content": text},
            ],
            n=1,
        )
        sentiment_text = completion.choices[0].message.content
        score = self.extract_float(sentiment_text)
        if score is None:
            raise ValueError(f"No floating score in llm. Output: {sentiment_text}")

        score_dict: dict[SentimentModel, float] = {
            SentimentModel.negative: abs(score - 1 / 2),
            SentimentModel.positive: abs(score + 1 / 2),
            SentimentModel.neutral: abs(score / 2),
        }

        return score_dict

    def extract_float(self, text: str) -> float | None:
        """Regex pattern to match float numbers"""
        pattern = r"[-+]?\d*\.?\d+([eE][-+]?\d+)?"

        # Search for the first match in the text
        match = re.search(pattern, text)

        score: float | None = float(match.group()) if match else None
        return score

    def preprocess(self, text: str) -> str:
        new_text = []
        for t in text.split(" "):
            t = "@user" if t.startswith("@") and len(t) > 1 else t
            t = "http" if t.startswith("http") else t
            new_text.append(t)
        return " ".join(new_text)

    def _average_scores(self, scores_dict: dict[SentimentModel, list[float]]):
        average_score_dict: dict[SentimentModel, float] = {}
        for sentiment, scores in scores_dict.items():
            average_score_dict[sentiment] = np.average(scores)

        return average_score_dict
