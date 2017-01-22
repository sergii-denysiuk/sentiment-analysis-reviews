import config
import utils

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.neighbors import KNeighborsClassifier

import pandas as pd


def random_forest(train_review_bag_of_words, train_review_sentiments, test_review_texts,
                  n_estimators=100):
    # initialize a Random Forest classifier with ``n_estimators`` trees
    forest = RandomForestClassifier(n_estimators=n_estimators)

    # fit the forest to the training set, using the bag of words as
    # features and the sentiment labels as the response variable
    forest = forest.fit(train_review_bag_of_words, train_review_sentiments)

    # use the random forest to make sentiment label predictions
    return forest.predict(test_review_texts)


def naive_bayes_gaussian(train_review_bag_of_words, train_review_sentiments, test_review_texts):
    # initialize a Naive Bayes Gaussian classifier
    nbg = GaussianNB()

    # fit the classifier to the training set, using the bag of words as
    # features and the sentiment labels as the response variable
    nbg = nbg.fit(train_review_bag_of_words, train_review_sentiments)

    # use the random forest to make sentiment label predictions
    return nbg.predict(test_review_texts)


def naive_bayes_multinomial(train_review_bag_of_words, train_review_sentiments, test_review_texts):
    # initialize a Naive Bayes Multinomial classifier
    nbm = MultinomialNB()

    # fit the classifier to the training set, using the bag of words as
    # features and the sentiment labels as the response variable
    nbm = nbm.fit(train_review_bag_of_words, train_review_sentiments)

    # use the random forest to make sentiment label predictions
    return nbm.predict(test_review_texts)


def naive_bayes_bernoulli(train_review_bag_of_words, train_review_sentiments, test_review_texts):
    # initialize a Naive Bayes Bernoulli classifier
    nbb = BernoulliNB()

    # fit the classifier to the training set, using the bag of words as
    # features and the sentiment labels as the response variable
    nbb = nbb.fit(train_review_bag_of_words, train_review_sentiments)

    # use the random forest to make sentiment label predictions
    return nbb.predict(test_review_texts)


def k_nearest_neighbors(train_review_bag_of_words, train_review_sentiments, test_review_texts,
                        n_neighbors=10, weights='uniform', algorithm='auto'):
    # initialize a k-Nnearest Neighbors classifier
    knn = KNeighborsClassifier(n_neighbors=n_neighbors, weights=weights, algorithm=algorithm)

    # fit the classifier to the training set, using the bag of words as
    # features and the sentiment labels as the response variable
    knn = knn.fit(train_review_bag_of_words, train_review_sentiments)

    # use the random forest to make sentiment label predictions
    return knn.predict(test_review_texts)


def write_results_to_csv(test_review_ids, test_reviews_sentiments_result,
                         test_review_sentiments, filename):
    # Copy the results to a pandas dataframe with an 'id' column and a 'sentiment' column
    output = pd.DataFrame(data={
        'id': test_review_ids,
        'sentiment_prediction': test_reviews_sentiments_result,
        'sentiment_actual': test_review_sentiments})

    # Use pandas to write the comma-separated output file
    output.to_csv(filename, index=False, quoting=3)


def run():
    print('Creating the bag of words...\n')

    vectorizer = CountVectorizer(analyzer='word',
                                 tokenizer=None,
                                 preprocessor=None,
                                 stop_words=None,
                                 max_features=5000)

    print('Cleaning and parsing the train set movie reviews...\n')

    train_pos_reviews = utils.read_and_parse(config.DATA_TRAINING_POS_REVIEW)
    train_neg_reviews = utils.read_and_parse(config.DATA_TRAINING_NEG_REVIEW)

    train_review_ids, train_review_texts, train_review_sentiments = utils.build_set(
        train_pos_reviews,
        train_neg_reviews,
        is_join=True,
        is_shuffle=True)

    # first: fits the model and learns the vocabulary
    # second: transforms our training data into feature vectors
    train_review_texts = vectorizer.fit_transform(train_review_texts).toarray()

    # shape of created model
    # print(train_review_texts.shape)

    # selected words
    # print(vectorizer.get_feature_names())

    # count of each word
    # print(utils.count_words(vectorizer.get_feature_names(), train_review_texts))

    print('Cleaning and parsing the test set movie reviews...\n')

    test_review_ids, test_review_texts, test_review_sentiments = utils.build_set(
        utils.read_and_parse(config.DATA_TEST_POS_REVIEW),
        utils.read_and_parse(config.DATA_TEST_NEG_REVIEW),
        is_join=True,
        is_shuffle=True)

    # get a bag of words for the test set, and convert to a numpy array
    test_review_texts = vectorizer.transform(test_review_texts).toarray()

    print('Training the Random Forest...\n')
    test_reviews_sentiments_result_rf = random_forest(
        train_review_texts, train_review_sentiments, test_review_texts)

    print('Training the Naive Bayes Gaussian...\n')
    test_reviews_sentiments_result_nbg = naive_bayes_gaussian(
        train_review_texts, train_review_sentiments, test_review_texts)

    print('Training the Naive Bayes Multinomial...\n')
    test_reviews_sentiments_result_nbm = naive_bayes_multinomial(
        train_review_texts, train_review_sentiments, test_review_texts)

    print('Training the Naive Bayes Bernoulli...\n')
    test_reviews_sentiments_result_nbb = naive_bayes_bernoulli(
        train_review_texts, train_review_sentiments, test_review_texts)

    print('Training the k-Nearest Neighbors...\n')
    test_reviews_sentiments_result_knn = k_nearest_neighbors(
        train_review_texts, train_review_sentiments, test_review_texts)

    print('Accuracy of the the Random Forest: {accuracy}\n'.format(
        accuracy=utils.calculate_accuracy(
            test_review_sentiments, test_reviews_sentiments_result_rf)))

    print('Accuracy of the Naive Bayes Gaussian: {accuracy}\n'.format(
        accuracy=utils.calculate_accuracy(
            test_review_sentiments, test_reviews_sentiments_result_nbg)))

    print('Accuracy of the Naive Bayes Multinomial: {accuracy}\n'.format(
        accuracy=utils.calculate_accuracy(
            test_review_sentiments, test_reviews_sentiments_result_nbm)))

    print('Accuracy of the Naive Bayes Bernoulli: {accuracy}\n'.format(
        accuracy=utils.calculate_accuracy(
            test_review_sentiments, test_reviews_sentiments_result_nbb)))

    print('Accuracy of the k-Nearest Neighbors: {accuracy}\n'.format(
        accuracy=utils.calculate_accuracy(
            test_review_sentiments, test_reviews_sentiments_result_knn)))

    rf_filename = 'bag-of-words-sklearn-rf-model.csv'
    nbg_filename = 'bag-of-words-sklearn-nbg-model.csv'
    nbm_filename = 'bag-of-words-sklearn-nbm-model.csv'
    nbb_filename = 'bag-of-words-sklearn-nbb-model.csv'
    knn_filename = 'bag-of-words-sklearn-knn-model.csv'

    print ('Wrote Random Forest results to {filename}\n'.format(filename=rf_filename))
    write_results_to_csv(
        test_review_ids, test_reviews_sentiments_result_rf,
        test_review_sentiments, rf_filename)

    print ('Wrote Naive Bayes Gaussian results to {filename}\n'.format(filename=nbg_filename))
    write_results_to_csv(
        test_review_ids, test_reviews_sentiments_result_nbg,
        test_review_sentiments, nbg_filename)

    print ('Wrote Naive Bayes Multinomial results to {filename}\n'.format(filename=nbg_filename))
    write_results_to_csv(
        test_review_ids, test_reviews_sentiments_result_nbm,
        test_review_sentiments, nbm_filename)

    print ('Wrote Naive Bayes Bernoulli results to {filename}\n'.format(filename=nbg_filename))
    write_results_to_csv(
        test_review_ids, test_reviews_sentiments_result_nbb,
        test_review_sentiments, nbb_filename)

    print ('Wrote k-Nearest Neighbors results to {filename}\n'.format(filename=nbg_filename))
    write_results_to_csv(
        test_review_ids, test_reviews_sentiments_result_knn,
        test_review_sentiments, knn_filename)


if __name__ == '__main__':
    try:
        run()
    except LookupError as e:
        import nltk
        nltk.download()
