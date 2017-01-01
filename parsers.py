import re
from bs4 import BeautifulSoup
from nltk.corpus import stopwords


class BaseParser(object):
    """
    Class with bacsic interface that must be implemented
    by inheritor parser class.
    """

    def __init__(self, text):
        """
        Initialize by some text.

        @args:
            text (string): raw text
        """
        self.text = text

    def parse(self):
        raise NotImplementedError("Should have implemented this")


class WordsParser(BaseParser):
    """
    Utility class for processing raw HTML text
    into segments for further learning.
    """

    def clean_html_markup(self, text, parser='html.parser'):
        """
        Remove HTML markup.

        @args:
            text (string)
        @kwargs:
            parser (string)
        @returns:
            string: text cleaned from HTML markup
        """
        return BeautifulSoup(text, parser).get_text()

    def clean_punctuation(self, text):
        """
        Remove non-letters.

        @args:
            text (string)
        @returns:
            string: text cleaned from numbers and other punctuation characters
        """
        return re.sub("[^a-zA-Z]", " ", text)

    def remove_stopwords(self, words, lang="english"):
        """
        Remove stop words

        @args:
            words (list)
        @kwargs:
            lang (string)
        @returns:
            list: words without stopwords
        """
        sw = set(stopwords.words(lang))
        return [w for w in words if w not in sw]

    def split_to_words(self, text):
        """
        Convert to lower case, split into individual words

        @args:
            text (string)
        @returns:
            list: list of words
        """
        return text.lower().split()

    def parse(self):
        """
        Get cleaned words from text.

        @returns:
            list: list with cleaned words
        """
        return self.remove_stopwords(
            self.split_to_words(
                self.clean_punctuation(
                    self.clean_html_markup(self.text))))
