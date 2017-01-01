import glob
import parsers
import random
import numpy


def read_and_parse(path_pattern, parser=parsers.WordsParser):
    """
    Read and clean data from all files that match to given path-pattern.

    @args:
        path_pattern (string): pattern for files that must be processed
    @kwargs:
        parser (class): parser class. Must be implementation of parsers.BaseParser class
    @returns:
        list: list, which items is a parsed and cleaned data from files
    """
    result = []

    for filename in glob.glob(path_pattern):
        with open(filename, 'r') as file:
            result.append(parser(file).parse())

    return result


def build_set(positive_set, negative_set,
              is_join=False, is_shuffle=False):
    """
    Build dataset from given positive and negative datasets.

    @args:
        positive_set (list): set with positive data
        negative_set (list): set with negative data
    @kwargs:
        is_join (boolean): is items in each set must be join from words to sentences
        is_shuffle (boolean): is positive and negative review must be shuffled
    @returns:
        tuple: tuple of two lists with reviews and it's sentiment values, respectively
    """
    reviews = []
    if is_join:
        reviews = [' '.join(i) for i in positive_set] + \
            [' '.join(i) for i in negative_set]
    else:
        reviews = positive_set + negative_set

    sentiments = [1] * len(positive_set) + [0] * len(negative_set)

    if is_shuffle:
        combined = list(zip(reviews, sentiments))
        random.shuffle(combined)

        reviews[:], sentiments[:] = zip(*combined)

    return (reviews, sentiments)


def count_words(words, dataset):
    """
    Count of each word from ``words`` in ``dataset``.

    @args:
        words (list): list of words to count
        dataset(list): list of lists with words to be counted
    @returns:
        list: list of tuples with word and the number of times it appears in the given dataset
    """
    result = []

    dist = numpy.sum(dataset, axis=0)

    for tag, count in zip(words, dist):
        result.append((count, tag))

    return result


def calculate_accuracy(actual_list, recieved_list):
    """
    Compare two lists and get percent of they matches.

    @args:
        actual_list (list): list of actual values
        recieved_list (list): list of received values
    @returns:
        float: percent of match values from recieved_list to values from actual_list
    """
    valid_part_len = 0

    for i, j in zip(actual_list, recieved_list):
        if i == j:
            valid_part_len += 1

    return (100.0 / len(actual_list)) * valid_part_len
