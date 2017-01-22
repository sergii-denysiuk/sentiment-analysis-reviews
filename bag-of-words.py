import config
import utils
import parsers
import classifiers.sklearn as classifiers_sk

from sklearn.feature_extraction.text import CountVectorizer


def run():
    print('Creating the bag of words...\n')

    vectorizer = CountVectorizer(analyzer='word',
                                 tokenizer=None,
                                 preprocessor=None,
                                 stop_words=None,
                                 max_features=5000)

    print('Cleaning and parsing the train set movie reviews...\n')

    train_pos = utils.read_and_parse(
        config.DATA_TRAINING_POS_REVIEW, parsers.WordsParser)
    train_neg = utils.read_and_parse(
        config.DATA_TRAINING_NEG_REVIEW, parsers.WordsParser)

    train_ids, train_texts, train_sentiments = utils.build_set(
        train_pos, train_neg,
        is_join=True, is_shuffle=True)

    # first: fits the model and learns the vocabulary
    # second: transforms our training data into feature vectors
    train_texts = vectorizer.fit_transform(train_texts).toarray()

    print('Cleaning and parsing the test set movie reviews...\n')

    test_pos = utils.read_and_parse(
        config.DATA_TEST_POS_REVIEW, parsers.WordsParser)
    test_neg = utils.read_and_parse(
        config.DATA_TEST_NEG_REVIEW, parsers.WordsParser)

    test_ids, test_texts, test_sentiments = utils.build_set(
        test_pos, test_neg,
        is_join=True, is_shuffle=True)

    # get a bag of words for the test set, and convert to a numpy array
    test_texts = vectorizer.transform(test_texts).toarray()

    print('Training the Random Forest...\n')
    test_sentiments_predicted_rf = classifiers_sk.random_forest(
        train_texts, train_sentiments, test_texts)

    print('Training the Naive Bayes Gaussian...\n')
    test_sentiments_predicted_nbg = classifiers_sk.naive_bayes_gaussian(
        train_texts, train_sentiments, test_texts)

    print('Training the Naive Bayes Multinomial...\n')
    test_sentiments_predicted_nbm = classifiers_sk.naive_bayes_multinomial(
        train_texts, train_sentiments, test_texts)

    print('Training the Naive Bayes Bernoulli...\n')
    test_sentiments_predicted_nbb = classifiers_sk.naive_bayes_bernoulli(
        train_texts, train_sentiments, test_texts)

    print('Training the k-Nearest Neighbors...\n')
    test_sentiments_predicted_knn = classifiers_sk.k_nearest_neighbors(
        train_texts, train_sentiments, test_texts)

    print('Accuracy of the the Random Forest: {accuracy}\n'.format(
        accuracy=utils.calculate_accuracy(
            test_sentiments, test_sentiments_predicted_rf)))

    print('Accuracy of the Naive Bayes Gaussian: {accuracy}\n'.format(
        accuracy=utils.calculate_accuracy(
            test_sentiments, test_sentiments_predicted_nbg)))

    print('Accuracy of the Naive Bayes Multinomial: {accuracy}\n'.format(
        accuracy=utils.calculate_accuracy(
            test_sentiments, test_sentiments_predicted_nbm)))

    print('Accuracy of the Naive Bayes Bernoulli: {accuracy}\n'.format(
        accuracy=utils.calculate_accuracy(
            test_sentiments, test_sentiments_predicted_nbb)))

    print('Accuracy of the k-Nearest Neighbors: {accuracy}\n'.format(
        accuracy=utils.calculate_accuracy(
            test_sentiments, test_sentiments_predicted_knn)))

    filename_rf = 'bag-of-words-sklearn-rf-model.csv'
    filename_nbg = 'bag-of-words-sklearn-nbg-model.csv'
    filename_nbm = 'bag-of-words-sklearn-nbm-model.csv'
    filename_nbb = 'bag-of-words-sklearn-nbb-model.csv'
    filename_knn = 'bag-of-words-sklearn-knn-model.csv'
    filename_summary = 'bag-of-words-sklearn-summary.txt'

    print('Wrote Random Forest results to {filename}\n'.format(
        filename=filename_rf))
    utils.write_results_to_csv(
        test_ids,
        test_sentiments,
        test_sentiments_predicted_rf,
        filename_rf)

    print('Wrote Naive Bayes Gaussian results to {filename}\n'.format(
        filename=filename_nbg))
    utils.write_results_to_csv(
        test_ids,
        test_sentiments,
        test_sentiments_predicted_nbg,
        filename_nbg)

    print('Wrote Naive Bayes Multinomial results to {filename}\n'.format(
        filename=filename_nbg))
    utils.write_results_to_csv(
        test_ids,
        test_sentiments,
        test_sentiments_predicted_nbm,
        filename_nbm)

    print('Wrote Naive Bayes Bernoulli results to {filename}\n'.format(
        filename=filename_nbg))
    utils.write_results_to_csv(
        test_ids,
        test_sentiments,
        test_sentiments_predicted_nbb,
        filename_nbb)

    print('Wrote k-Nearest Neighbors results to {filename}\n'.format(
        filename=filename_nbg))
    utils.write_results_to_csv(
        test_ids,
        test_sentiments,
        test_sentiments_predicted_knn,
        filename_knn)

    with open(filename_summary, "w") as file_summary:
        print('Size of train dataset: {size}'.format(
            size=len(train_ids)), file=file_summary)

        print('Size of test dataset: {size}'.format(
            size=len(test_ids)), file=file_summary)

        print('\n', file=file_summary)

        print('Accuracy of the the Random Forest: {accuracy}'.format(
            accuracy=utils.calculate_accuracy(
                test_sentiments, test_sentiments_predicted_rf)), file=file_summary)

        print('Accuracy of the Naive Bayes Gaussian: {accuracy}'.format(
            accuracy=utils.calculate_accuracy(
                test_sentiments, test_sentiments_predicted_nbg)), file=file_summary)

        print('Accuracy of the Naive Bayes Multinomial: {accuracy}'.format(
            accuracy=utils.calculate_accuracy(
                test_sentiments, test_sentiments_predicted_nbm)), file=file_summary)

        print('Accuracy of the Naive Bayes Bernoulli: {accuracy}'.format(
            accuracy=utils.calculate_accuracy(
                test_sentiments, test_sentiments_predicted_nbb)), file=file_summary)

        print('Accuracy of the k-Nearest Neighbors: {accuracy}'.format(
            accuracy=utils.calculate_accuracy(
                test_sentiments, test_sentiments_predicted_knn)), file=file_summary)

        print('\n', file=file_summary)

        print('Count of each word in train dataset: {counts}'.format(
            counts=utils.count_words(vectorizer.get_feature_names(), train_texts)), file=file_summary)


if __name__ == '__main__':
    run()
