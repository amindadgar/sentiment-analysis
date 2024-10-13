from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer, AutoConfig
import numpy as np
from scipy.special import softmax
from schema import SentimentModel
from hezar.models import Model


class AnalyzeSentiment:
    def __init__(self) -> None:
        MODEL = f"cardiffnlp/twitter-roberta-base-sentiment-latest"

        self.tokenizer = AutoTokenizer.from_pretrained(MODEL)
        self.config = AutoConfig.from_pretrained(MODEL)
        self.model = AutoModelForSequenceClassification.from_pretrained(MODEL)
        self.hezarai_model = Model.load("hezarai/bert-fa-sentiment-dksf")

    def process_twitter_roberta(self, texts: list[str]) -> dict[SentimentModel, float]:
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
            message_sentiments = self.process_twitter_roberta(message)
            for sentiment, score in message_sentiments.items():
                scores_dict[sentiment].append(score)

        # averaging the score
        for sentiment in scores_dict:
            scores_dict[sentiment] = np.average(scores_dict[sentiment])

        return scores_dict

    def _twitter_roberta_process(self, text: str) -> dict[SentimentModel, float]:
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
        output = self.model(**encoded_input)
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

        # Averaging
        for sentiment in scores_dict.keys():
            scores_dict[sentiment] = np.average(scores_dict[sentiment])

        return scores_dict

    def preprocess(self, text: str) -> str:
        new_text = []
        for t in text.split(" "):
            t = "@user" if t.startswith("@") and len(t) > 1 else t
            t = "http" if t.startswith("http") else t
            new_text.append(t)
        return " ".join(new_text)
