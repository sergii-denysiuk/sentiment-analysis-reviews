import config
import utils
import parsers
import classifiers.sklearn as classifiers_sk

from gensim.models import Word2Vec
import numpy as np


def get_avg_feature_vectors(reviews, model, num_features):
    """
    Given a set of reviews (each one a list of words), calculate
    the average feature vector for each one and return a 2D numpy array
    """

    # initialize a counter
    counter = 0.

    # preallocate a 2D numpy array, for speed
    reviewfeature_vecs = np.zeros(
        (len(reviews), num_features), dtype='float32')

    # loop through the reviews
    for review in reviews:
        # call the function that makes average feature vectors
        reviewfeature_vecs[counter] = make_feature_vec(
            review, model, num_features)

        # increment the counter
        counter += 1.

    return reviewfeature_vecs


def make_feature_vec(words, model, num_features):
    """
    Function to average all of the word vectors in a given paragraph
    """

    # pre-initialize an empty numpy array (for speed)
    feature_vec = np.zeros((num_features,), dtype='float32')

    nwords = 0.

    # index2word is a list that contains the names of the words in
    # the model's vocabulary. Convert it to a set, for speed
    index2wordset = set(model.index2word)

    # loop over each word in the review and, if it is in the model's
    # vocaublary, add its feature vector to the total
    for word in words:
        if word in index2wordset:
            nwords += 1.
            feature_vec = np.add(feature_vec, model[word])

    # divide the result by the number of words to get the average
    feature_vec = np.divide(feature_vec, nwords)
    return feature_vec


def run():
    print('Cleaning and parsing the train set movie reviews as sentences...\n')

    train_sentences_pos = utils.read_and_parse(
        config.DATA_TRAINING_POS_REVIEW, parser=parsers.SentencesParser)
    train_sentences_neg = utils.read_and_parse(
        config.DATA_TRAINING_NEG_REVIEW, parser=parsers.SentencesParser)

    train_sentences_ids, train_sentences_texts, train_sentences_sentiments = utils.build_set(
        train_sentences_pos,
        train_sentences_neg,
        is_join=False,
        is_shuffle=False)

    # convert list or reviews which is list of sentences which is list words (three-level nested list)
    # to list of sentences which list of words (two-level nested list)
    sentences = [y for x in train_sentences_texts for y in x]

    print('Cleaning and parsing the train set movie reviews as words...\n')

    train_words_pos = utils.read_and_parse(
        config.DATA_TRAINING_POS_REVIEW)
    train_words_neg = utils.read_and_parse(
        config.DATA_TRAINING_NEG_REVIEW)

    train_words_ids, train_words_texts, train_words_sentiments = utils.build_set(
        train_words_pos,
        train_words_neg,
        is_join=False,
        is_shuffle=False)

    print('Cleaning and parsing the test set movie reviews as words...\n')

    test_words_pos = utils.read_and_parse(
        config.DATA_TEST_POS_REVIEW)
    test_words_neg = utils.read_and_parse(
        config.DATA_TEST_NEG_REVIEW)

    test_words_ids, test_words_texts, test_words_sentiments = utils.build_set(
        test_words_pos,
        test_words_neg,
        is_join=False,
        is_shuffle=True)

    # initialize and train the model (this will take some time)
    print('Training Word2Vec model...\n')

    # set values for various parameters
    num_features = 300  # word vector dimensionality
    min_word_count = 40  # minimum word count
    num_workers = 4  # number of threads to run in parallel
    context = 10  # context window size
    downsampling = 1e-3  # downsample setting for frequent words

    model = Word2Vec(sentences, workers=num_workers,
                     size=num_features, min_count=min_word_count,
                     window=context, sample=downsampling, seed=1)

    # if you don't plan to train the model any further, calling
    # init_sims will make the model much more memory-efficient.
    model.init_sims(replace=True)

    # it can be helpful to create a meaningful model name and
    # save the model for later use. You can load it later using Word2Vec.load()
    model_name = 'word-to-vector-model'
    model.save(model_name)

    print('Creating average feature vecs for training reviews')

    train_words_vectors = get_avg_feature_vectors(
        train_words_texts, model, num_features)

    print('Creating average feature vecs for test reviews')

    test_words_vectors = get_avg_feature_vectors(
        test_words_texts, model, num_features)

    print('Training the Random Forest, then make predictions...\n')
    test_words_sentiments_predicted_rf = classifiers_sk.random_forest(
        train_words_vectors, train_words_sentiments, test_words_vectors)

    print('Training the Naive Bayes Gaussian, then make predictions...\n')
    test_words_sentiments_predicted_nbg = classifiers_sk.naive_bayes_gaussian(
        train_words_vectors, train_words_sentiments, test_words_vectors)

    print('Training the Naive Bayes Bernoulli, then make predictions...\n')
    test_words_sentiments_predicted_nbb = classifiers_sk.naive_bayes_bernoulli(
        train_words_vectors, train_words_sentiments, test_words_vectors)

    print('Training the k-Nearest Neighbors, then make predictions...\n')
    test_words_sentiments_predicted_knn = classifiers_sk.k_nearest_neighbors(
        train_words_vectors, train_words_sentiments, test_words_vectors)

    print('Accuracy of the the Random Forest: {accuracy}\n'.format(
        accuracy=utils.calculate_accuracy(
            test_words_sentiments, test_words_sentiments_predicted_rf)))

    print('Accuracy of the Naive Bayes Gaussian: {accuracy}\n'.format(
        accuracy=utils.calculate_accuracy(
            test_words_sentiments, test_words_sentiments_predicted_nbg)))

    print('Accuracy of the Naive Bayes Bernoulli: {accuracy}\n'.format(
        accuracy=utils.calculate_accuracy(
            test_words_sentiments, test_words_sentiments_predicted_nbb)))

    print('Accuracy of the k-Nearest Neighbors: {accuracy}\n'.format(
        accuracy=utils.calculate_accuracy(
            test_words_sentiments, test_words_sentiments_predicted_knn)))

    filename_rf = 'word-to-vector-bag-of-centroids-rf-model.csv'
    filename_nbg = 'word-to-vector-bag-of-centroids-nbg-model.csv'
    filename_nbb = 'word-to-vector-bag-of-centroids-nbb-model.csv'
    filename_knn = 'word-to-vector-bag-of-centroids-knn-model.csv'
    filename_summary = 'word-to-vector-bag-of-centroids-summary.txt'

    print('Wrote Random Forest results to {filename}\n'.format(
        filename=filename_rf))
    utils.write_results_to_csv(
        test_words_ids,
        test_words_sentiments,
        test_words_sentiments_predicted_rf,
        filename_rf)

    print('Wrote Naive Bayes Gaussian results to {filename}\n'.format(
        filename=filename_nbg))
    utils.write_results_to_csv(
        test_words_ids,
        test_words_sentiments,
        test_words_sentiments_predicted_nbg,
        filename_nbg)

    print('Wrote Naive Bayes Bernoulli results to {filename}\n'.format(
        filename=filename_nbg))
    utils.write_results_to_csv(
        test_words_ids,
        test_words_sentiments,
        test_words_sentiments_predicted_nbb,
        filename_nbb)

    print('Wrote k-Nearest Neighbors results to {filename}\n'.format(
        filename=filename_nbg))
    utils.write_results_to_csv(
        test_words_ids,
        test_words_sentiments,
        test_words_sentiments_predicted_knn,
        filename_knn)

    with open(filename_summary, "w") as file_summary:
        print('Size of train dataset: {size}'.format(
            size=len(train_words_ids)), file=file_summary)

        print('Size of test dataset: {size}'.format(
            size=len(test_words_ids)), file=file_summary)

        print('\n', file=file_summary)

        print('Accuracy of the the Random Forest: {accuracy}'.format(
            accuracy=utils.calculate_accuracy(
                test_words_sentiments, test_words_sentiments_predicted_rf)), file=file_summary)

        print('Accuracy of the Naive Bayes Gaussian: {accuracy}'.format(
            accuracy=utils.calculate_accuracy(
                test_words_sentiments, test_words_sentiments_predicted_nbg)), file=file_summary)

        print('Accuracy of the Naive Bayes Bernoulli: {accuracy}'.format(
            accuracy=utils.calculate_accuracy(
                test_words_sentiments, test_words_sentiments_predicted_nbb)), file=file_summary)

        print('Accuracy of the k-Nearest Neighbors: {accuracy}'.format(
            accuracy=utils.calculate_accuracy(
                test_words_sentiments, test_words_sentiments_predicted_knn)), file=file_summary)


if __name__ == '__main__':
    run()
