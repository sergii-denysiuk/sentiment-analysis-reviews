
class AverageVectors:
    """
    Vector Averaging.

    One challenge with the IMDB dataset is the variable-length reviews.
    We need to find a way to take individual word vectors
    and transform them into a feature set that is the same length for every review.

    Since each word is a vector in 300-dimensional space,
    we can use vector operations to combine the words in each review.
    One method we tried was to simply average the word vectors
    in a given review (for this purpose, we removed stop words, which would just add noise).
    """

    @classmethod
    def get(cls, train_words_texts, test_words_texts, model, num_features):
        """
        Return train and test average vectors.
        """

        # example result:
        # [[-0.01175839, -0.02443665, 0.03473418, ...],
        #  [-0.01756256, -0.00818371, 0.01030297, ...], ...]
        train_words_vectors = cls.get_avg_feature_vectors(
            train_words_texts, model, num_features)

        # example result:
        # [[-0.01175839, -0.02443665, 0.03473418, ...],
        #  [-0.01756256, -0.00818371, 0.01030297, ...], ...]
        test_words_vectors = cls.get_avg_feature_vectors(
            test_words_texts, model, num_features)

        return train_words_vectors, test_words_vectors

    @classmethod
    def get_avg_feature_vectors(cls, reviews, model, num_features):
        """
        Given a set of reviews (each one a list of words),
        calculate, the average feature vector for each one,
        and return a 2D numpy array (list of lists).
        """

        # initialize a counter
        counter = 0.

        # preallocate a 2D numpy array, for speed
        review_feature_vecs = np.zeros(
            (len(reviews), num_features), dtype='float32')

        # loop through the reviews
        for review in reviews:
            # call the function that makes average feature vectors
            review_feature_vecs[counter] = cls.make_feature_vec(
                review, model, num_features)
            # increment the counter
            counter += 1.

        return review_feature_vecs

    @classmethod
    def make_feature_vec(cls, words, model, num_features):
        """
        Function to average all of the word vectors in a given paragraph.
        """

        # pre-initialize an empty numpy array (for speed)
        feature_vec = np.zeros(
            (num_features,), dtype='float32')

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
