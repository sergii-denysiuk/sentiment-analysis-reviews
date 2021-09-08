from abc import ABCMeta
import re

from bs4 import BeautifulSoup
import nltk.data

try:
    nltk.data.find('stopwords')
except LookupError:
    nltk.download('stopwords')

from nltk.corpus import stopwords


EN_STOPWORDS = set(stopwords.words('english'))


class BaseParser(metaclass=ABCMeta):

    @classmethod
    def clean_html_markup(cls, text, parser='html.parser'):
        """Remove HTML markup."""
        return BeautifulSoup(text, parser).get_text()

    @classmethod
    def remove_non_letters(cls, text):
        """Remove non-letters."""
        return re.sub('[^a-zA-Z]', ' ', text)

    @classmethod
    def to_lower(cls, text):
        """Convert text to lowercase."""
        return text.lower()

    @classmethod
    def split_to_words(cls, text):
        """Split into individual words."""
        return text.split()

    @classmethod
    def remove_stopwords(cls, words, stopwords_list=EN_STOPWORDS):
        """Remove stopwords."""
        return [w for w in words if w not in EN_STOPWORDS]

    @classmethod
    def parse(cls, text, *args, **kwargs):
        raise NotImplementedError("Should have implemented this")
