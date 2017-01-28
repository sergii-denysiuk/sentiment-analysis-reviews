import glob
import parsers
import random
import numpy
import pandas


def read_and_parse(path_pattern, parser=parsers.WordsParser, **kwargs):
    """
    Read and clean data from all files that match to given path-pattern.

    @args:
        path_pattern (string): pattern for files that must be processed
    @kwargs:
        parser (class): parser class. Must be implementation of parsers.BaseParser class
    @returns:
        list: list of tuples, which items is a parsed and cleaned data from files
              example [('file', 'review text'), ...])
    """
    result = []

    for filename in glob.glob(path_pattern):
        with open(filename, 'r') as file:
            result.append((filename, parser(file).parse(**kwargs)))

    return result


def concate_sets(positive_set, negative_set,
                 is_join=False, is_shuffle=False):
    """
    Build dataset from given positive and negative datasets.

    @args:
        positive_set (list): set with positive data
                             example [('file_1', 'positive review text'), ...])
        negative_set (list): set with negative data
                             example [('file_2', 'negative review text'), ...])
    @kwargs:
        is_join (boolean): is items in each set must be join from words to sentences
        is_shuffle (boolean): is positive and negative review must be shuffled
    @returns:
        tuple: tuple of two lists with reviews and it's sentiment values, respectively
    """
    reviews_ids = []
    reviews_texts = []
    reviews_sentiments = []

    reviews_ids = [i[0] for i in positive_set] + \
        [i[0] for i in negative_set]

    if is_join:
        reviews_texts = [' '.join(i[1]) for i in positive_set] + \
            [' '.join(i[1]) for i in negative_set]
    else:
        reviews_texts = [i[1] for i in positive_set] + \
            [i[1] for i in negative_set]

    reviews_sentiments = [1] * len(positive_set) + [0] * len(negative_set)

    if is_shuffle:
        combined = list(zip(reviews_ids, reviews_texts, reviews_sentiments))
        random.shuffle(combined)
        reviews_ids[:], reviews_texts[:], reviews_sentiments[:] = zip(*combined)

    return (reviews_ids, reviews_texts, reviews_sentiments)


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


def calculate_accuracy(actual_list, predicted_list):
    """
    Compare two lists and get percent of they matches.

    @args:
        actual_list (list): list of actual values
        predicted_list (list): list of received values
    @returns:
        float: percent of match values from predicted_list to values from actual_list
    """
    valid_part_len = 0

    for i, j in zip(actual_list, predicted_list):
        if i == j:
            valid_part_len += 1

    return (100.0 / len(actual_list)) * valid_part_len


def write_results_to_csv(ids, sentiments_actuals,
                         sentiments_predictions, filename):
    """
    Write the results to a pandas dataframe.

    @args:
        sentiments_actuals (list): list of actual sentiments
        sentiments_predictions (list): list of predictions for sentiments
        filename (string): name of file to write in
    """
    output = pandas.DataFrame(data={
        'id': ids,
        'sentiment_actual': sentiments_actuals,
        'sentiment_prediction': sentiments_predictions})

    # use pandas to write the comma-separated output file
    output.to_csv(filename, index=False, quoting=3)
