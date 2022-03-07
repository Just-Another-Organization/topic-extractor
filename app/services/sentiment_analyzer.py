import gensim
import nltk
from gensim.models import LdaMulticore
from gensim.test.utils import common_corpus, common_dictionary
from nltk.corpus import stopwords

from utils.logger import Logger

nltk.download('stopwords')


class TopicsExtractor:
    LANG = 'english'

    def __init__(self):
        self.logger = Logger('TopicsExtractor')
        self.stopwords = stopwords.words(self.LANG)
        self.stopwords.extend(['from', 'subject', 're', 'edu', 'use'])
        self.lda = LdaMulticore(common_corpus, id2word=common_dictionary, num_topics=10)

    def remove_stopwords(self, text):
        words = []
        for word in text:
            if word not in self.stopwords:
                words.append(word)
        return words

    def preprocess(self, text):
        # deacc =True removes punctuations
        words = gensim.utils.simple_preprocess(text, deacc=True)
        return self.remove_stopwords(words)

    def print_topics(self):
        x = self.lda.show_topics(num_topics=12, num_words=5, formatted=False)
        topics_words = [(tp[0], [wd[0] for wd in tp[1]]) for tp in x]

        # Below Code Prints Topics and Words
        for topic, words in topics_words:
            print(str(topic) + "::" + str(words))

        # Below Code Prints Only Words
        for topic, words in topics_words:
            print(" ".join(words))

    def extract_topics(self, text):
        words = self.preprocess(text)
        other_corpus = common_dictionary.doc2bow(words)
        self.lda.update(other_corpus)
        self.print_topics()
