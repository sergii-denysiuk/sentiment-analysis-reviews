import config
import utils

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier

import pandas as pd


def run():
    print('Cleaning and parsing the train set movie reviews...\n')

    train_pos_reviews = utils.read_and_parse(config.DATA_TRAINING_POS_REVIEW)
    train_neg_reviews = utils.read_and_parse(config.DATA_TRAINING_NEG_REVIEW)

    train_review_ids, train_review_texts, train_review_sentiments = utils.build_set(
        train_pos_reviews,
        train_neg_reviews,
        is_join=True,
        is_shuffle=True)

    print('Creating the bag of words...\n')

    vectorizer = CountVectorizer(analyzer='word',
                                 tokenizer=None,
                                 preprocessor=None,
                                 stop_words=None,
                                 max_features=5000)

    # first: fits the model and learns the vocabulary
    # second: transforms our training data into feature vectors
    train_review_texts = vectorizer.fit_transform(train_review_texts).toarray()

    # shape of created model
    # print(train_review_texts.shape)

    # selected words
    # print(vectorizer.get_feature_names())

    # count of each word
    # print(utils.count_words(vectorizer.get_feature_names(), train_review_texts))

    print('Training the random forest...')

    # initialize a Random Forest classifier with 100 trees
    forest = RandomForestClassifier(n_estimators=100)

    # fit the forest to the training set, using the bag of words as
    # features and the sentiment labels as the response variable
    forest = forest.fit(train_review_texts, train_review_sentiments)

    print('Cleaning and parsing the test set movie reviews...\n')

    test_review_ids, test_review_texts, test_review_sentiments = utils.build_set(
        utils.read_and_parse(config.DATA_TEST_POS_REVIEW),
        utils.read_and_parse(config.DATA_TEST_NEG_REVIEW),
        is_join=True,
        is_shuffle=True)

    # get a bag of words for the test set, and convert to a numpy array
    test_review_texts = vectorizer.transform(test_review_texts).toarray()

    # use the random forest to make sentiment label predictions
    test_reviews_sentiments_result = forest.predict(test_review_texts)

    print('Accuracy :', utils.calculate_accuracy(
        test_review_sentiments, test_reviews_sentiments_result))

    filename = 'bag-of-words-model.csv'
    print ('Wrote results to {filename}'.format(filename=filename))

    # Copy the results to a pandas dataframe with an 'id' column and a 'sentiment' column
    output = pd.DataFrame(data={
        'id': test_review_ids,
        'sentiment_prediction': test_reviews_sentiments_result,
        'sentiment_actual': test_review_sentiments})

    # Use pandas to write the comma-separated output file
    output.to_csv(filename, index=False, quoting=3)


if __name__ == '__main__':
    try:
        run()
    except LookupError as e:
        import nltk
        nltk.download()
