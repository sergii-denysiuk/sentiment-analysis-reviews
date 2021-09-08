"""
 The Bag of Words model learns a vocabulary text's, then models each text by counting the number of times each word appears. For example, consider the following two sentences:
Sentence 1: "The cat sat on the hat"
Sentence 2: "The dog ate the cat and the hat"

From these two sentences, vocabulary is as follows:
{ the, cat, sat, on, hat, dog, ate, and }

To get bags of words, count the number of times each word occurs in each sentence.
Sentence 1: { 2, 1, 1, 1, 1, 0, 0, 0 }
Sentence 2: { 3, 1, 0, 0, 1, 1, 1, 1 }

In the IMDB data, there is a very large number of reviews, which will give a large vocabulary. To limit the size of the feature vectors, choose some maximum vocabulary size. Below, used the 5000 most frequent words (remembering that stop words have already been removed).
"""

import config
import parsers
import utils
import classifiers as classifiers_sk

from sklearn.feature_extraction.text import CountVectorizer


def run():
    print("Read train data...")

    train_data = utils.concat_sets(
        utils.read_and_parse(config.DATA_TRAINING_POS_REVIEW, parsers.WordsParser),
        utils.read_and_parse(config.DATA_TRAINING_NEG_REVIEW, parsers.WordsParser),
        is_join=True, is_shuffle=True)

    print("Read test data...")

    test_data = utils.concat_sets(
        utils.read_and_parse(config.DATA_TEST_POS_REVIEW, parsers.WordsParser),
        utils.read_and_parse(config.DATA_TEST_NEG_REVIEW, parsers.WordsParser),
        is_join=True, is_shuffle=True)

    print('Creating the bag of words...')

    # note that CountVectorizer comes with its own options
    # to automatically do preprocessing, tokenization, and stop word removal
    # for each of these, instead of specifying "None",
    # it's possible to use a built-in method or custom function,
    # however, in this example, for data cleaning used custom parsers
    vectorizer = CountVectorizer(analyzer='word',
                                 tokenizer=None,
                                 preprocessor=None,
                                 stop_words=None,
                                 max_features=5000)

    print('Cleaning and parsing the train set movie reviews...')

    # get a bag of words for the training set, and convert to a numpy array
    # example result:
    # train_texts -> [[1, 3], [1, 2], [3, 1], ...]
    train_texts = vectorizer.fit_transform(train_texts).toarray()

    print('Cleaning and parsing the test set movie reviews...')

    # get a bag of words for the test set, and convert to a numpy array
    # example result:
    # test_texts -> [[1, 3], [1, 2], [3, 1], ...]
    test_texts = vectorizer.transform(test_texts).toarray()

    print('Training the Random Forest...')
    n_estimators = 100
    # example result:
    # test_sentiments_predicted_rf -> [1, 0, 1...]
    test_sentiments_predicted_rf = classifiers_sk.random_forest(
        train_texts, train_sentiments, test_texts, n_estimators=n_estimators)

    print('Training the Naive Bayes Gaussian...')
    # example result:
    # test_sentiments_predicted_nbg -> [1, 0, 1...]
    test_sentiments_predicted_nbg = classifiers_sk.naive_bayes_gaussian(
        train_texts, train_sentiments, test_texts)

    print('Training the Naive Bayes Multinomial...')
    # example result:
    # test_sentiments_predicted_nbm -> [1, 0, 1...]
    test_sentiments_predicted_nbm = classifiers_sk.naive_bayes_multinomial(
        train_texts, train_sentiments, test_texts)

    print('Training the Naive Bayes Bernoulli...')
    # example result:
    # test_sentiments_predicted_nbb -> [1, 0, 1...]
    test_sentiments_predicted_nbb = classifiers_sk.naive_bayes_bernoulli(
        train_texts, train_sentiments, test_texts)

    print('Training the k-Nearest Neighbors...')
    n_neighbors = 100
    # example result:
    # test_sentiments_predicted_knn -> [1, 0, 1...]
    test_sentiments_predicted_knn = classifiers_sk.k_nearest_neighbors(
        train_texts, train_sentiments, test_texts, n_neighbors=n_neighbors)

    print('Accuracy of the the Random Forest: {accuracy}'.format(
        accuracy=utils.calculate_accuracy(
            test_sentiments, test_sentiments_predicted_rf)))

    print('Accuracy of the Naive Bayes Gaussian: {accuracy}'.format(
        accuracy=utils.calculate_accuracy(
            test_sentiments, test_sentiments_predicted_nbg)))

    print('Accuracy of the Naive Bayes Multinomial: {accuracy}'.format(
        accuracy=utils.calculate_accuracy(
            test_sentiments, test_sentiments_predicted_nbm)))

    print('Accuracy of the Naive Bayes Bernoulli: {accuracy}'.format(
        accuracy=utils.calculate_accuracy(
            test_sentiments, test_sentiments_predicted_nbb)))

    print('Accuracy of the k-Nearest Neighbors: {accuracy}'.format(
        accuracy=utils.calculate_accuracy(
            test_sentiments, test_sentiments_predicted_knn)))

    filename_sklearn_rf = 'bag-of-words-sklearn-rf-model.csv'
    filename_sklearn_nbg = 'bag-of-words-sklearn-nbg-model.csv'
    filename_sklearn_nbm = 'bag-of-words-sklearn-nbm-model.csv'
    filename_sklearn_nbb = 'bag-of-words-sklearn-nbb-model.csv'
    filename_sklearn_knn = 'bag-of-words-sklearn-knn-model.csv'
    filename_summary = 'bag-of-words-summary.txt'

    print('Wrote Random Forest results to {filename}'.format(
        filename=filename_sklearn_rf))
    utils.write_results_to_csv(
        test_ids,
        test_sentiments,
        test_sentiments_predicted_rf,
        filename_sklearn_rf)

    print('Wrote Naive Bayes Gaussian results to {filename}'.format(
        filename=filename_sklearn_nbg))
    utils.write_results_to_csv(
        test_ids,
        test_sentiments,
        test_sentiments_predicted_nbg,
        filename_sklearn_nbg)

    print('Wrote Naive Bayes Multinomial results to {filename}'.format(
        filename=filename_sklearn_nbm))
    utils.write_results_to_csv(
        test_ids,
        test_sentiments,
        test_sentiments_predicted_nbm,
        filename_sklearn_nbm)

    print('Wrote Naive Bayes Bernoulli results to {filename}'.format(
        filename=filename_sklearn_nbb))
    utils.write_results_to_csv(
        test_ids,
        test_sentiments,
        test_sentiments_predicted_nbb,
        filename_sklearn_nbb)

    print('Wrote k-Nearest Neighbors results to {filename}'.format(
        filename=filename_sklearn_knn))
    utils.write_results_to_csv(
        test_ids,
        test_sentiments,
        test_sentiments_predicted_knn,
        filename_sklearn_knn)

    print('Wrote summary results to {filename}'.format(
        filename=filename_summary))

    with open(filename_summary, "w") as file_summary:
        print('Size of train dataset: {size}'.format(
            size=len(train_ids)), file=file_summary)

        print('Size of test dataset: {size}'.format(
            size=len(test_ids)), file=file_summary)

        print('\n', file=file_summary)

        print('Number of trees in Random Forest: {trees}'.format(
            trees=n_estimators), file=file_summary)

        print('Number of neighbors in KNN: {neighbors}'.format(
            neighbors=n_neighbors), file=file_summary)

        print('\n', file=file_summary)

        print('Accuracy of the the Random Forest sklearn: {accuracy}'.format(
            accuracy=utils.calculate_accuracy(
                test_sentiments, test_sentiments_predicted_rf)), file=file_summary)

        print('Accuracy of the Naive Bayes Gaussian sklearn: {accuracy}'.format(
            accuracy=utils.calculate_accuracy(
                test_sentiments, test_sentiments_predicted_nbg)), file=file_summary)

        print('Accuracy of the Naive Bayes Multinomial sklearn: {accuracy}'.format(
            accuracy=utils.calculate_accuracy(
                test_sentiments, test_sentiments_predicted_nbm)), file=file_summary)

        print('Accuracy of the Naive Bayes Bernoulli sklearn: {accuracy}'.format(
            accuracy=utils.calculate_accuracy(
                test_sentiments, test_sentiments_predicted_nbb)), file=file_summary)

        print('Accuracy of the k-Nearest Neighbors sklearn: {accuracy}'.format(
            accuracy=utils.calculate_accuracy(
                test_sentiments, test_sentiments_predicted_knn)), file=file_summary)

        print('\n', file=file_summary)

        print('Count of each word in train dataset: {counts}'.format(
            counts=utils.count_words(vectorizer.get_feature_names(), train_texts)), file=file_summary)


if __name__ == '__main__':
    run()
