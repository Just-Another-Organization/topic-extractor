import os

from keybert import KeyBERT
from transformers import pipeline

from utils.logger import Logger


class Analyzer:
    MODEL_LANG = os.getenv('MODEL_LANG', 'english')
    MODEL_NAME = os.getenv('MODEL_NAME', 'all-MiniLM-L6-v2')
    CLASSIFIER_NAME = os.getenv('CLASSIFIER_NAME', 'facebook/bart-large-mnli')
    CLASSIFIER_TYPE = os.getenv('CLASSIFIER_TYPE', 'zero-shot-classification')

    def __init__(self):
        self.logger = Logger('TopicsExtractor')

        self.logger.info('MODEL_LANG: ' + self.MODEL_LANG)
        self.logger.info('MODEL_NAME: ' + self.MODEL_NAME)
        self.logger.info('CLASSIFIER_NAME: ' + self.CLASSIFIER_NAME)
        self.logger.info('CLASSIFIER_TYPE: ' + self.CLASSIFIER_TYPE)

        # KeyBERT model can be changed: https://maartengr.github.io/KeyBERT/guides/embeddings.html
        self.model = KeyBERT(self.MODEL_NAME)
        self.classifier = pipeline(self.CLASSIFIER_TYPE, model=self.CLASSIFIER_NAME)

    def extract_keywords(self, text, top_n=10):
        keywords = self.model.extract_keywords(
            text,
            keyphrase_ngram_range=(1, 1),
            stop_words=self.MODEL_LANG,
            highlight=False,
            top_n=top_n)
        keywords_list = list(dict(keywords).keys())
        return keywords_list

    def classify_text_labels(self, text, labels, multi=True):
        return self.classifier(text, labels, multi_label=multi)
