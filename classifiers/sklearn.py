from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.neighbors import KNeighborsClassifier


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
                        n_neighbors=100, weights='uniform', algorithm='auto'):
    # initialize a k-Nnearest Neighbors classifier
    knn = KNeighborsClassifier(
        n_neighbors=n_neighbors, weights=weights, algorithm=algorithm)

    # fit the classifier to the training set, using the bag of words as
    # features and the sentiment labels as the response variable
    knn = knn.fit(train_review_bag_of_words, train_review_sentiments)

    # use the random forest to make sentiment label predictions
    return knn.predict(test_review_texts)
