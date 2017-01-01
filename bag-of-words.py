import config
import utils

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier


def run():
    print("Cleaning and parsing the train set movie reviews...\n")
    train_reviews, train_sentiments = utils.build_set(
        utils.read_and_parse(config.DATA_TRAINING_POS_REVIEW),
        utils.read_and_parse(config.DATA_TRAINING_NEG_REVIEW),
        is_join=True,
        is_shuffle=True)

    print("Creating the bag of words...\n")
    vectorizer = CountVectorizer(analyzer="word",
                                 tokenizer=None,
                                 preprocessor=None,
                                 stop_words=None,
                                 max_features=5000)

    # first: fits the model and learns the vocabulary
    # second: transforms our training data into feature vectors
    train_reviews = vectorizer.fit_transform(train_reviews).toarray()

    # shape of created model
    # print(train_reviews.shape)

    # selected words
    # print(vectorizer.get_feature_names())

    # count of each word
    # print(utils.count_words(vectorizer.get_feature_names(), train_reviews))

    print("Training the random forest...")
    # initialize a Random Forest classifier with 100 trees
    forest = RandomForestClassifier(n_estimators=100)

    # fit the forest to the training set, using the bag of words as
    # features and the sentiment labels as the response variable
    forest = forest.fit(train_reviews, train_sentiments)

    print("Cleaning and parsing the test set movie reviews...\n")
    test_reviews, test_sentiments = utils.build_set(
        utils.read_and_parse(config.DATA_TEST_POS_REVIEW),
        utils.read_and_parse(config.DATA_TEST_NEG_REVIEW),
        is_join=True,
        is_shuffle=True)

    # get a bag of words for the test set, and convert to a numpy array
    test_reviews = vectorizer.transform(test_reviews).toarray()

    # use the random forest to make sentiment label predictions
    test_sentiments_result = forest.predict(test_reviews)

    print("Accuracy :", utils.calculate_accuracy(
        test_sentiments, test_sentiments_result))


if __name__ == '__main__':
    try:
        run()
    except LookupError as e:
        import nltk
        nltk.download()
