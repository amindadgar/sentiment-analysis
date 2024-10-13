import re


class TextCleaner:
    def __init__(self) -> None:
        pass

    def remove_emoji(self, text: str) -> str:
        emoji_pattern = re.compile(
            "["
            "\U0001F600-\U0001F64F"  # Emoticons
            "\U0001F300-\U0001F5FF"  # Symbols & Pictographs
            "\U0001F680-\U0001F6FF"  # Transport & Map Symbols
            "\U0001F1E0-\U0001F1FF"  # Flags (iOS)
            "\U00002702-\U000027B0"  # Dingbats
            "\U000024C2-\U0001F251"  # Enclosed characters
            "\u200c"  # Zero Width Non-Joiner
            "\u200d"  # Zero Width Joiner
            "\u200f"
            "]+",
            flags=re.UNICODE,
        )
        return emoji_pattern.sub(r"", text)
