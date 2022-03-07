from keybert import KeyBERT
from transformers import pipeline

from utils.logger import Logger


class Analyzer:
    LANG = 'english'
    CLASSIFIER = 'facebook/bart-large-mnli'

    def __init__(self):
        self.logger = Logger('TopicsExtractor')
        # Default model can be changed: https://maartengr.github.io/KeyBERT/guides/embeddings.html
        self.model = KeyBERT()
        self.classifier = pipeline("zero-shot-classification", model=self.CLASSIFIER)

    def extract_keywords(self, text, top_n=10):
        keywords = self.model.extract_keywords(
            text,
            keyphrase_ngram_range=(1, 1),
            stop_words=self.LANG,
            highlight=False,
            top_n=top_n)
        keywords_list = list(dict(keywords).keys())
        return keywords_list

    def classify_text_labels(self, text, labels, multi=True):
        return self.classifier(text, labels, multi_label=multi)
