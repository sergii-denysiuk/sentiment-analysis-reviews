import typing

from parsers import base as base_parser


def read_and_parse(path_pattern: str,
                   parser: typing.Type[base_parser.BaseParser],
                   **kwargs: typing.Any) -> Iterator[typing.Tuple[str, str]]:
    """
    Read and clean data from all files that match to given path-pattern.

    :param path_pattern: pattern for files that must be processed
    :param parser: parser class
    :return: tuple with filename and parsed and cleaned data from file, example ('file', file_data)
    """
    for filename in glob.glob(path_pattern):
        with open(filename, 'r') as file:
            yield filename, parser.parse(file.read(), **kwargs)


def concat_sets(positive_reviews: typing.Iterable[typing.Tuple[str, str]],
                negative_reviews: typing.Iterable[typing.Tuple[str, str]],
                columns typing.List[str],
                is_join: bool = False,
                is_shuffle: bool = False) -> pd.DataFrame:
    """
    Build dataset from given positive and negative datasets.

    :param positive_reviews: list with positive data, example [('file_1', 'positive review text'), ...])
    :param negative_reviews: list with negative data, example [('file_2', 'negative review text'), ...])
    :param columns: columns names for returned DataFrame
    :param is_join: join items in each set from words to sentences
    :param is_shuffle: shuffle positive and negative reviews
    :return: table with filenames, reviews and it's sentiment values, respectively
    """
    data = []

    for reviews, sentiment in ((positive_reviews, True),
                               (negative_reviews, False)):
        data.extend((filename,
                     " ".join(filedata) if is_join else filedata,
                     sentiment)
                    for filename, filedata in reviews)

    if is_shuffle:
        random.shuffle(data)

    return pd.DataFrame(data,
                        columns=columns)
