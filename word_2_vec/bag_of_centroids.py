
class BagOfCentroids:
    """
    Word2Vec creates clusters of semantically related words,
    so another possible approach is to exploit the similarity of words
    within a cluster. Grouping vectors in this way is known as "vector quantization."
    To accomplish this, we first need to find the centers of the word clusters,
    which we can do by using a clustering algorithm such as K-Means.

    In K-Means, the one parameter we need to set is "K," or the number of clusters.
    How should we decide how many clusters to create?
    Trial and error suggested that small clusters,
    with an average of only 5 words or so per cluster,
    gave better results than large clusters with many words
    """

    @classmethod
    def get(cls, train_words_texts, test_words_texts, model, num_features):
        """
        Return train and test average vectors.
        """
        print('Run k-means on the word vectors and print a few clusters...')

        start = time.time()

        # set 'k' (num_clusters) to be 1/5th of the vocabulary size
        # or an average of 5 words per cluster
        word_vectors = model.syn0
        num_clusters = int(word_vectors.shape[0] / 5)

        # initalize a k-means object and use it to extract centroids
        kmeans_clustering = KMeans(n_clusters=num_clusters)
        idx = kmeans_clustering.fit_predict(word_vectors)

        end = time.time()
        elapsed = end - start

        print('Time taken for K Means clustering: {time} seconds.'.format(
            time=elapsed))

        # create a Word: Index dictionary, mapping each vocabulary word to a cluster number
        word_centroid_map = dict(zip(model.index2word, idx))

        print('Show first ten clusters:')

        for cluster_i in range(0, 10):
            # find all of the words for that cluster number
            words = []
            for i in range(0, len(word_centroid_map.values())):
                if(list(word_centroid_map.values())[i] == cluster_i):
                    words.append(list(word_centroid_map.keys())[i])

            # eample output:
            # $ Cluster number 0
            # $ Cluster words: [u'passport', u'penthouse', u'suite', u'seattle', u'apple']
            print('Cluster number: {number}. \n Cluster words: {words} \n\n'.format(
                number=cluster_i,
                words=words))

        print('Create bags of centroids...')

        # pre-allocate an array for the training and test sets bags of centroids (for speed)
        train_centroids = np.zeros(
            (len(train_words_texts), num_clusters),
            dtype='float32')
        test_centroids = np.zeros(
            (len(test_words_texts), num_clusters),
            dtype='float32')

        # transform the training and test sets reviews into bags of centroids
        for index, review in enumerate(train_words_texts):
            train_centroids[index] = cls.create_bag_of_centroids(
                review, word_centroid_map)

        for index, review in enumerate(test_words_texts):
            test_centroids[index] = cls.create_bag_of_centroids(
                review, word_centroid_map)

        return (train_centroids, test_centroids)

    @classmethod
    def create_bag_of_centroids(cls, wordlist, word_centroid_map):
        """
        Convert review into bags-of-centroids.
        """

        # the number of clusters is equal to the highest cluster index in the `word_centroid_map``
        num_centroids = max(word_centroid_map.values()) + 1

        # pre-allocate the bag of centroids vector (for speed)
        bag_of_centroids = np.zeros(num_centroids, dtype='float32')

        # loop over the words in the review. If the word is in the vocabulary,
        # find which cluster it belongs to, and increment that cluster count by one
        for word in wordlist:
            if word in word_centroid_map:
                index = word_centroid_map[word]
                bag_of_centroids[index] += 1

        # return the 'bag of centroids'
        return bag_of_centroids
