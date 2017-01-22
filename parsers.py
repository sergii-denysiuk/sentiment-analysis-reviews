import re
from bs4 import BeautifulSoup

import nltk.data

try:
    nltk.data.find('stopwords')
    nltk.data.find('tokenizers')
except LookupError:
    nltk.download('stopwords')
    nltk.download('punkt')

from nltk.corpus import stopwords
stopwords_list = set(stopwords.words('english'))
tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')


class BaseParser(object):
    """
    Class with bacsic interface that must be implemented
    by inheritor parser class.
    """

    def __init__(self, file):
        """
        Initialize by some text.

        @args:
            file (_io.TextIOWrapper): file with raw text
        """
        self.file = file

    def clean_html_markup(self, file, parser='html.parser'):
        """
        Remove HTML markup.

        @args:
            file (_io.TextIOWrapper): file with raw text
        @kwargs:
            parser (string)
        @returns:
            string: text cleaned from HTML markup
        """
        return BeautifulSoup(file, parser).get_text()

    def remove_non_letters(self, text):
        """
        Remove non-letters.

        @args:
            text (string)
        @returns:
            string: text cleaned from numbers and other punctuation characters
        """
        return re.sub("[^a-zA-Z]", " ", text)

    def split_to_words(self, text):
        """
        Split into individual words.

        @args:
            text (string)
        @returns:
            list: list of words
        """
        return text.split()

    def split_to_sentences(self, text, tokenizer=tokenizer):
        """
        Split into individual sentences.

        @args:
            text (string)
        @returns:
            list: list of sentences
        """
        return tokenizer.tokenize(text.strip())

    def to_lower(self, text):
        """
        Convert text to lowercase.

        @args:
            text (string)
        @returns:
            string: text
        """
        return text.lower()

    def remove_stopwords(self, words, stopwords_list=stopwords_list):
        """
        Remove stop words

        @args:
            words (list)
        @kwargs:
            lang (string)
        @returns:
            list: words without stopwords
        """
        return [w for w in words if w not in stopwords_list]

    def parse(self):
        raise NotImplementedError("Should have implemented this")


class WordsParser(BaseParser):
    """
    Utility class for processing raw HTML text
    into segments of words for further learning.
    """

    def parse(self,
              is_remove_non_letters=True,
              is_to_lower=True,
              is_remove_stopwords=True):
        """
        Get cleaned words from text.

        @kwargs:
            is_remove_non_letters (boolean): is remove non-letters
            is_to_lower (boolean): is convert to lowercase
            is_remove_stopwords (boolean): is remove the stopwords
        @returns:
            list: list with cleaned words
        """
        result = self.clean_html_markup(self.file)

        if is_remove_non_letters:
            result = self.remove_non_letters(result)

        if is_to_lower:
            result = self.to_lower(result)

        result = self.split_to_words(result)

        if is_remove_stopwords:
            result = self.remove_stopwords(result)

        return result


class SentencesParser(BaseParser):
    """
    Utility class for processing raw HTML text
    into segments of sentences for further learning.
    """

    def parse(self,
              tokenizer=tokenizer,
              is_remove_non_letters=True,
              is_to_lower=True,
              is_sentence_split_to_words=True,
              is_remove_stopwords=False):
        """
        Split a text into parsed sentences.text
        It is better not to remove stop words because
        the algorithm relies on the broader context of the sentence
        in order to produce high-quality word vectors.

        @kwargs:
            tokenizer (object): tokenizer to split the paragraph into sentences
            is_remove_stopwords (boolean): is to remove stopwords
        @returns:
            list: list of sentences. Each sentence is a list of words, so this returns a list of lists.
        """
        sentences = []
        result = self.clean_html_markup(self.file)
        raw_sentences = self.split_to_sentences(result, tokenizer)

        for raw_sentence in raw_sentences:
            if len(raw_sentence) > 0:
                if is_remove_non_letters:
                    raw_sentence = self.remove_non_letters(raw_sentence)

                if is_to_lower:
                    raw_sentence = self.to_lower(raw_sentence)

                if is_sentence_split_to_words:
                    raw_sentence = self.split_to_words(raw_sentence)

                if is_remove_stopwords:
                    raw_sentence = self.remove_stopwords(raw_sentence)

                sentences.append(raw_sentence)

        return sentences
