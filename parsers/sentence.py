import nltk.data

try:
    nltk.data.find('tokenizers')
except LookupError:
    nltk.download('punkt')

from parsers.base import BaseParser


EN_TOKENIZER = nltk.data.load('tokenizers/punkt/english.pickle')


class SentencesParser(BaseParser):
    """
    Implement processing raw HTML text
    into segments of sentences for further learning.
    """

    @classmethod
    def split_to_sentences(cls, text, tokenizer):
        """Split into individual sentences."""
        return tokenizer.tokenize(text.strip())

    @classmethod
    def parse(cls,
              text,
              tokenizer=EN_TOKENIZER,
              is_sentence_split_to_words=True,
              is_remove_non_letters=True,
              is_remove_stopwords=False):
        """
        Split a text into parsed sentences.
        It is better not to remove stop words because
        the algorithm relies on the broader context of the sentence
        in order to produce high-quality word vectors.

        :param text: text to parse
        :type text: string
        :param tokenizer: tokenizer to split the paragraph into sentences
        :type tokenizer: object
        :param is_sentence_split_to_words: does sencences must be splited to words
        :type is_sentence_split_to_words: bool
        :param is_remove_non_letters: does non-letters have to be removed
        :type is_remove_non_letters: bool
        :param is_remove_stopwords: does stopwords have to be removed
        :type is_remove_stopwords: bool
        :return: list of sentences. Each sentence can be splited to a list of words, so this returns a list of lists
        :rtype: list
        """
        sentences = []
        result = cls.clean_html_markup(text)
        raw_sentences = cls.split_to_sentences(result, tokenizer)

        for raw_sentence in raw_sentences:
            if len(raw_sentence) > 0:
                raw_sentence = cls.to_lower(raw_sentence)

                if is_remove_non_letters:
                    raw_sentence = cls.remove_non_letters(raw_sentence)

                if is_sentence_split_to_words:
                    raw_sentence = cls.split_to_words(raw_sentence)

                if is_remove_stopwords:
                    raw_sentence = cls.remove_stopwords(raw_sentence)

                sentences.append(raw_sentence)

        return sentences
