import json
import logging

from tqdm import tqdm

from .text_cleaner import TextCleaner


class Processor:
    def __init__(self) -> None:
        self.cleaner = TextCleaner()

    def process_telegram_json(self, file_name: str) -> dict[str, list[str]]:
        """
        Process telegram json export file
        how to export: https://telegram.org/blog/export-and-more

        Parameters
        --------------
        file_name : str
            the json file

        Returns
        ---------
        data : dict[str, list[str]]
            each user name is a key of the dictionary
            and their messages is in the dictionary values
        """
        with open(file_name) as file:
            data = json.load(file)

        data_raw = {}

        for user_msgs in tqdm(data["chats"]["list"]):
            try:
                name = user_msgs["name"]
                messages = user_msgs["messages"]

                if name == "Telegram":
                    raise ValueError("Telegram messages shouldn't be processed!")

                plain_messages = []

                for msgs in messages:
                    for entity in msgs["text_entities"]:
                        if entity["type"] == "plain":
                            plain_messages.append(
                                self.cleaner.remove_emoji(entity["text"])
                            )

                data_raw[name] = plain_messages

            except Exception as exp:
                logging.error(f"Error: {exp}!")

        return data_raw
