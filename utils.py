import glob
import random
import numpy as np
import pandas as pd


def read_and_parse(path_pattern, parser, **kwargs):
    """
    Read and clean data from all files that match to given path-pattern.

    :param path_pattern: pattern for files that must be processed
    :type path_pattern: string
    :param parser: parser class. Must be implementation of parsers.BaseParser class
    :type parser: object
    :return: tuple with filename and parsed and cleaned data from file
             example ('file', file_data)
    :rtype: Iterator[:class:`tuple`]
    """
    for filename in glob.glob(path_pattern):
        with open(filename, 'r') as file:
            yield filename, parser.parse(file.read(), **kwargs)


def concat_sets(positive_reviews, negative_reviews, columns,
                is_join=False, is_shuffle=False):
    """
    Build dataset from given positive and negative datasets.

    :param positive_reviews: list with positive data
                         example [('file_1', 'positive review text'), ...])
    :type positive_reviews: Iterable
    :param negative_reviews: list with negative data
                         example [('file_2', 'negative review text'), ...])
    :type negative_reviews: Iterable
    :param columns: columns names for returned DataFrame
    :type columns: list
    :param is_join: join items in each set from words to sentences
    :type is_join: bool
    :param is_shuffle: shuffle positive and negative reviews
    :type is_shuffle: bool
    :return : table with filenames, reviews and it's sentiment values, respectively
    :rtype: pd.DataFrame
    """
    data = []

    for reviews, sentiment in ((positive_reviews, True),
                               (negative_reviews, False)):
        data.extend((filename,
                     ' '.join(filedata) if is_join else filedata,
                     sentiment)
                    for filename, filedata in reviews)

    if is_shuffle:
        random.shuffle(data)

    return pd.DataFrame(data,
                        columns=columns)


def count_words(words, dataset):
    """
    Count of each word from `words` in `dataset`.

    :param words: words to count
    :type words: list
    :param dataset: lists of words to be counted
    :type dataset: list of lists
    :return: list of tuples with word and the number of times it appeared in the given dataset
    :rtype: list
    """
    result = []

    dist = np.sum(dataset, axis=0)

    for tag, count in zip(words, dist):
        result.append((count, tag))

    return result


def calculate_accuracy(actual_list, predicted_list):
    """
    Compare two lists and get percent of they matches.

    :param actual_list: actual values
    :type actual_list: list
    :param predicted_list: received values
    :type predicted_list: list
    :return: percent of match values from predicted_list to values from actual_list
    :rtype: float
    """
    valid_part_len = sum(i == j for i, j in zip(actual_list, predicted_list))
    return (100.0 / len(actual_list)) * valid_part_len


def write_results_to_csv(ids,
                         sentiments_actuals,
                         sentiments_predictions,
                         filename):
    """
    Write the results to a pandas dataframe.

    :param sentiments_actuals: list of actual sentiments
    :type sentiments_actuals: list
    :param sentiments_predictions: list of predictions for sentiments
    :type sentiments_predictions: list
    :param filename: name of file to write in
    :type filename: string
    """
    output = pd.DataFrame(data={
        "id": ids,
        "sentiment_actual": sentiments_actuals,
        "sentiment_predicted": sentiments_predictions})
    output.to_csv(filename, index=False, quoting=3)
