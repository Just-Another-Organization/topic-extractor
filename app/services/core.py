from services.analyzer import Analyzer
from utils.logger import Logger


class Core:

    def __init__(self):
        self.logger = Logger('Core')
        self.analyzer = Analyzer()

    def extract_keywords(self, text, top_n):
        return self.analyzer.extract_keywords(text, top_n)

    def classify_text_labels(self, text, labels, multi):
        result = []
        classification = self.analyzer.classify_text_labels(text, labels, multi)
        for i, label in enumerate(classification['labels']):
            score = classification['scores'][i]
            result.append({
                label: score
            })
        return result
