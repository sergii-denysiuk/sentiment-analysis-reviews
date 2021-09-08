from parsers.base import BaseParser


class WordsParser(BaseParser):
    """
    Implement processing raw HTML text
    into segments of words for further learning.
    """

    @classmethod
    def parse(cls,
              text,
              is_remove_non_letters=True,
              is_remove_stopwords=True):
        """
        Get cleaned words from text.

        :param text: text to parse
        :type text: string
        :param is_remove_non_letters: does non-letters have to be removed
        :type is_remove_non_letters: bool
        :param is_remove_stopwords: does stopwords have to be removed
        :type is_remove_stopwords: bool
        :return: list with cleaned words
        :rtype: list
        """
        result = cls.clean_html_markup(text)

        if is_remove_non_letters:
            result = cls.remove_non_letters(result)

        result = cls.to_lower(result)
        result = cls.split_to_words(result)

        if is_remove_stopwords:
            result = cls.remove_stopwords(result)

        return result
