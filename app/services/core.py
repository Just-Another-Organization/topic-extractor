from services.sentiment_analyzer import TopicsExtractor
from utils.logger import Logger


class Core:

    def __init__(self):
        self.logger = Logger('Core')
        self.extractor = TopicsExtractor()

    def extract_topics(self, text):
        topics = self.extractor.extract_topics(text)
        return topics
