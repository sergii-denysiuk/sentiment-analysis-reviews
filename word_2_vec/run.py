"""
Word2vec, published by Google in 2013, is a neural network implementation that learns distributed representations for words. The original code is in C, but it has since been ported to other languages, including Python.

Word2Vec does not need labels in order to create meaningful representations. This is useful, since most data in the real world is unlabeled. If the network is given enough training data (tens of billions of words), it produces word vectors with intriguing characteristics. Words with similar meanings appear in clusters, and clusters are spaced such that some word relationships, such as analogies, can be reproduced using vector math. The famous example is that, with highly trained word vectors, "king - man + woman = queen."

Although Word2Vec does not require graphics processing units (GPUs) like many deep learning algorithms, it is compute intensive. Both Google's version and the Python version rely on multi-threading (running multiple processes in parallel on your computer to save time). ln order to train your model in a reasonable amount of time, you will need to install cython. Word2Vec will run without cython installed, but it will take days to run instead of minutes.
"""

from gensim.models import Word2Vec
from sklearn.cluster import KMeans
import numpy as np
import time

import config
import parsers
import utils
import classifiers as classifiers_sk


def predict_and_save(train_ids, train_reviews, train_sentiments,
                     test_ids, test_reviews, test_sentiments,
                     filename_sklearn_rf, filename_sklearn_nbg,
                     filename_sklearn_nbb, filename_sklearn_knn,
                     filename_summary):
    print('Training the Random Forest, then make predictions...')
    n_estimators = 100
    # example result:
    # test_sentiments_predicted_rf -> [1, 0, 1...]
    test_sentiments_predicted_rf = classifiers_sk.random_forest(
        train_reviews, train_sentiments, test_reviews, n_estimators=n_estimators)

    print('Training the Naive Bayes Gaussian, then make predictions...')
    # example result:
    # test_sentiments_predicted_nbg -> [1, 0, 1...]
    test_sentiments_predicted_nbg = classifiers_sk.naive_bayes_gaussian(
        train_reviews, train_sentiments, test_reviews)

    print('Training the Naive Bayes Bernoulli, then make predictions...')
    # example result:
    # test_sentiments_predicted_nbb -> [1, 0, 1...]
    test_sentiments_predicted_nbb = classifiers_sk.naive_bayes_bernoulli(
        train_reviews, train_sentiments, test_reviews)

    print('Training the k-Nearest Neighbors, then make predictions...')
    n_neighbors = 100
    # example result:
    # test_sentiments_predicted_knn -> [1, 0, 1...]
    test_sentiments_predicted_knn = classifiers_sk.k_nearest_neighbors(
        train_reviews, train_sentiments, test_reviews, n_neighbors=n_neighbors)

    print('\n')

    print('Accuracy of the the Random Forest: {accuracy}'.format(
        accuracy=utils.calculate_accuracy(
            test_sentiments, test_sentiments_predicted_rf)))

    print('Accuracy of the Naive Bayes Gaussian: {accuracy}'.format(
        accuracy=utils.calculate_accuracy(
            test_sentiments, test_sentiments_predicted_nbg)))

    print('Accuracy of the Naive Bayes Bernoulli: {accuracy}'.format(
        accuracy=utils.calculate_accuracy(
            test_sentiments, test_sentiments_predicted_nbb)))

    print('Accuracy of the k-Nearest Neighbors: {accuracy}'.format(
        accuracy=utils.calculate_accuracy(
            test_sentiments, test_sentiments_predicted_knn)))

    print('\n')

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

    with open(filename_summary, 'w') as file_summary:
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

        print('Accuracy of the the Random Forest: {accuracy}'.format(
            accuracy=utils.calculate_accuracy(
                test_sentiments, test_sentiments_predicted_rf)), file=file_summary)

        print('Accuracy of the Naive Bayes Gaussian: {accuracy}'.format(
            accuracy=utils.calculate_accuracy(
                test_sentiments, test_sentiments_predicted_nbg)), file=file_summary)

        print('Accuracy of the Naive Bayes Bernoulli: {accuracy}'.format(
            accuracy=utils.calculate_accuracy(
                test_sentiments, test_sentiments_predicted_nbb)), file=file_summary)

        print('Accuracy of the k-Nearest Neighbors: {accuracy}'.format(
            accuracy=utils.calculate_accuracy(
                test_sentiments, test_sentiments_predicted_knn)), file=file_summary)


def run():
    print('Cleaning and parsing the train set movie reviews as sentences...')

    # example result:
    # train_sentences_ids -> [1, 2, ...]
    # train_sentences_texts -> [[['s1_word1', 's1_word2'], ['s2_word3', 's2_word4']],
    #                           [['s3_word5', 's3_word6'], ['s4_word7', 's4_word8']], ...]
    # train_sentences_sentiments -> [1, 0, ...]
    train_sentences_ids, train_sentences_texts, train_sentences_sentiments = utils.concat_sets(
        utils.read_and_parse(config.DATA_TRAINING_POS_REVIEW, parsers.SentencesParser),
        utils.read_and_parse(config.DATA_TRAINING_NEG_REVIEW, parsers.SentencesParser),
        is_join=False,
        is_shuffle=False)

    # Word2Vec expects single sentences, each one as a list of words.
    # in other words, the input format is a list of lists
    # convert list of reviews which is list of sentences which is list words (three-level nested list)
    # to list of sentences which list of words (two-level nested list)
    # example result:
    # train_sentences_texts -> [['s1_word1', 's1_word2'], ['s2_word3', 's2_word4'],
    #                           ['s3_word5', 's3_word6'], ['s4_word7', 's4_word8'], ...]
    sentences = [y for x in train_sentences_texts for y in x]

    print('Cleaning and parsing the train set movie reviews as words...')

    # example result:
    # train_ids -> [1, 2, 3, ...]
    # train_texts -> [['word1', 'word2'], ['word3', 'word4'], ['word5', 'wor6'], ...]
    # train_sentiments -> [1, 0, 0, ...]
    train_words_ids, train_words_texts, train_words_sentiments = utils.concat_sets(
        utils.read_and_parse(config.DATA_TRAINING_POS_REVIEW),
        utils.read_and_parse(config.DATA_TRAINING_NEG_REVIEW),
        is_join=False, is_shuffle=False)

    print('Cleaning and parsing the test set movie reviews as words...')

    # example result:
    # train_ids -> [1, 2, 3, ...]
    # train_texts -> [['word1', 'word2'], ['word3', 'word4'], ['word5', 'wor6'], ...]
    # train_sentiments -> [1, 0, 0, ...]
    test_words_ids, test_words_texts, test_words_sentiments = utils.concat_sets(
        utils.read_and_parse(config.DATA_TEST_POS_REVIEW),
        utils.read_and_parse(config.DATA_TEST_NEG_REVIEW),
        is_join=False, is_shuffle=True)

    # initialize and train the model (this will take some time)
    print('Training Word2Vec model...')

    # word vector dimensionality - more features result in longer runtimes,
    # and often, but not always, result in better models,
    # reasonable values can be in the tens to hundreds
    num_features = 300

    # minimum word count - this helps limit the size of the vocabulary to meaningful words.
    # Any word that does not occur at least this many times across all documents is ignored.
    # Reasonable values could be between 10 and 100.
    # In this case, since each movie occurs 30 times, we set the minimum word count to 40,
    # to avoid attaching too much importance to individual movie titles.
    # This resulted in an overall vocabulary size of around 15,000 words.
    # Higher values also help limit run time.
    min_word_count = 40

    # number of threads to run in parallel - number of parallel processes to run.
    # This is computer-specific, but between 4 and 6 should work on most
    # systems.
    num_workers = 4

    # context window size - how many words of context should the training algorithm take into account?
    # 10 seems to work well for hierarchical softmax (more is better, up to a
    # point).
    context = 10

    # downsample setting for frequent words - the Google documentation recommends
    # values between .00001 and .001.
    # For us, values closer 0.001 seemed to improve the accuracy of the final
    # model.
    downsampling = 1e-3

    model = Word2Vec(sentences, workers=num_workers,
                     size=num_features, min_count=min_word_count,
                     window=context, sample=downsampling, seed=1)

    # if you don't plan to train the model any further, calling
    # init_sims will make the model much more memory-efficient.
    model.init_sims(replace=True)

    # it can be helpful to create a meaningful model name and
    # save the model for later use. You can load it later using Word2Vec.load()
    # model_name = 'word-2-vec-average-vectors-model'
    # model.save(model_name)
    # model = Word2Vec.load(model_name)

    print('\n')

    print('Example of model usage:')

    # try to deduce which word in a set is most dissimilar from the others:
    print(model.doesnt_match('man woman child kitchen'.split()))
    print(model.doesnt_match('france england germany berlin'.split()))
    print(model.doesnt_match('paris berlin london austria'.split()))

    # get insight into the model's word clusters
    print(model.most_similar('man'))
    print(model.most_similar('queen'))
    print(model.most_similar('awful'))

    print('End of example of model usage.')

    print('\n')

    print('Creating average feature vectors for training and test reviews...')
    average_vectors_train_words, average_vectors_test_words = AverageVectors.get(
        train_words_texts, test_words_texts, model, num_features)

    av_filename_sklearn_rf = 'word-2-vec-average-vectors-sklearn-rf-model.csv'
    av_filename_sklearn_nbg = 'word-2-vec-average-vectors-sklearn-nbg-model.csv'
    av_filename_sklearn_nbb = 'word-2-vec-average-vectors-sklearn-nbb-model.csv'
    av_filename_sklearn_knn = 'word-2-vec-average-vectors-sklearn-knn-model.csv'
    av_filename_summary = 'word-2-vec-average-vectors-summary.txt'

    print('Predicting by using average feature vectors...')

    predict_and_save(train_words_ids, average_vectors_train_words, train_words_sentiments,
                     test_words_ids, average_vectors_test_words, test_words_sentiments,
                     av_filename_sklearn_rf, av_filename_sklearn_nbg,
                     av_filename_sklearn_nbb, av_filename_sklearn_knn,
                     av_filename_summary)

    print('\n')

    print('Creating bag of centroids for training and test reviews...')
    bag_of_centroids_train_words, bag_of_centroids_test_words = BagOfCentroids.get(
        train_words_texts, test_words_texts, model, num_features)

    boc_filename_sklearn_rf = 'word-2-vec-bag-of-centroids-sklearn-rf-model.csv'
    boc_filename_sklearn_nbg = 'word-2-vec-bag-of-centroids-sklearn-nbg-model.csv'
    boc_filename_sklearn_nbb = 'word-2-vec-bag-of-centroids-sklearn-nbb-model.csv'
    boc_filename_sklearn_knn = 'word-2-vec-bag-of-centroids-sklearn-knn-model.csv'
    boc_filename_summary = 'word-2-vec-bag-of-centroids-summary.txt'

    print('Predicting by using bag of centroids...')

    predict_and_save(train_words_ids, bag_of_centroids_train_words, train_words_sentiments,
                     test_words_ids, bag_of_centroids_test_words, test_words_sentiments,
                     boc_filename_sklearn_rf, boc_filename_sklearn_nbg,
                     boc_filename_sklearn_nbb, boc_filename_sklearn_knn,
                     boc_filename_summary)


if __name__ == '__main__':
    run()
