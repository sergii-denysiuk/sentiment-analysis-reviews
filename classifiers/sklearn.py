from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.neighbors import KNeighborsClassifier


def random_forest(train_review_bag_of_words, train_review_sentiments,
                  n_estimators=100):
    """Random Forest classifier."""
    forest = RandomForestClassifier(n_estimators=n_estimators)
    forest = forest.fit(train_review_bag_of_words, train_review_sentiments)
    return forest


def naive_bayes_gaussian(train_review_bag_of_words, train_review_sentiments):
    """Naive Bayes Gaussian classifier."""
    nbg = GaussianNB()
    nbg = nbg.fit(train_review_bag_of_words, train_review_sentiments)
    return nbg


def naive_bayes_multinomial(train_review_bag_of_words, train_review_sentiments):
    """Naive Bayes Multinomial classifier."""
    nbm = MultinomialNB()
    nbm = nbm.fit(train_review_bag_of_words, train_review_sentiments)
    return nbm


def naive_bayes_bernoulli(train_review_bag_of_words, train_review_sentiments):
    """Naive Bayes Bernoulli classifier."""
    nbb = BernoulliNB()
    nbb = nbb.fit(train_review_bag_of_words, train_review_sentiments)
    return nbb


def k_nearest_neighbors(train_review_bag_of_words, train_review_sentiments,
                        n_neighbors=100, weights='uniform', algorithm='auto'):
    """k-Nnearest Neighbors classifier."""
    knn = KNeighborsClassifier(
        n_neighbors=n_neighbors, weights=weights, algorithm=algorithm)
    knn = knn.fit(train_review_bag_of_words, train_review_sentiments)
    return knn
