# Introduction

Sentiment analysis is a challenging subject in machine learning. People express their emotions in language that is often obscured by sarcasm, ambiguity, and plays on words, all of which could be very misleading for both humans and computers. This project is an example of sentiment analysis for movie review. It's divided into parts:
* Part 1. Basic Natural Language Processing: how to use **BagOfWords** model for sentiment analysis.
* Part 2. Deep Learning for Text Understanding: how to train a model using **Word2Vec** and how to use the resulting word vectors for sentiment analysis.


# Tools

This project is implemented in Python 3.* programming language. It is use basic NLP techniques from `nltk` library and machine learning classifiers from `scikit-learn` library, such as:
* Random Forest
* Naive Bayes Gaussian
* Naive Bayes Multinomial
* Naive Bayes Bernoulli
* k-Nearest Neighbors

**BagOfWords** model implemented by usung `scikit-learn` library.

**Word2Vec** implemented by using `gensim` library. In order to train **Word2Vec** model in a reasonable amount of time, it's required to install `cython`. **Word2Vec** will run without `cython` installed, but it will take days to run instead of minutes.

For text cleaning, was used the *Beautiful Soup* library.

After finishing all calculations, for saving all results into a file `pandas` library is used. As a helper for working with arrays `numpy` library was used.


# Data

To achieve these goals, used a set of 25,000 highly polar movie reviews for training, and 25,000 for testing. There is additional unlabeled data for use as well. This dataset was collected in association with the following publication: http://ai.stanford.edu/~amaas/data/sentiment/


# Theory

## BagOfWords
The BagOfWords model is a simplifying representation used in natural language processing and information retrieval (IR). In this model, a text (such as a sentence or a document) is represented as the bag (multiset) of its words, disregarding grammar and even word order but keeping multiplicity.

## Word2Vec
Google's Word2Vec is a deep-learning inspired method that focuses on the meaning of words. Word2Vec attempts to understand meaning and semantic relationships among words. It works in a way that is similar to deep approaches, such as recurrent neural nets or deep neural nets, but is computationally more efficient.

Word2Vec does not need labels in order to create meaningful representations. This is useful, since most data in the real world is unlabeled. If the network is given enough training data (tens of billions of words), it produces word vectors with intriguing characteristics. Words with similar meanings appear in clusters, and clusters are spaced such that some word relationships, such as analogies, can be reproduced using vector math. The famous example is that, with highly trained word vectors, "king - man + woman = queen."


# Run project

Create virtual environment and install all dependencies from `requirements.txt`:
```bash
mkvirtualenv -p /usr/bin/python3 virualenv_name
workon virualenv_name
pip install -r requirements.txt
```

To run **BagOfWords** model, type:
```bash
python /path/to/project/bag_of_words/run.py
```

To run **Word2Vec** model, type:
```bash
python /path/to/project/word_2_vec/run.py
```


# Description of experiments

## BagOfWords
- reading the test and training data
- data cleaning and text preprocessing of test and training data
    - removing HTML markup (`BeautifulSoup`)
    - dealing with punctuation and numbers (`Regular expressions`)
    - convert to lower case
    - split into individual words (called "tokenization" in `NLP` lingo)
    - dealing with stopwords (`NLTK`)
- creating features from a BagOfWords (using `CountVectorizer` from `scikit-learn`)
- train classifiers
- make a predictions
- calculate accuracy of predictions and save the results

## Word2Vec
- reading the data for training of Word2Vec model
- data cleaning and text preprocessing of training data for Word2Vec
    - removing HTML markup (`BeautifulSoup`)
    - split to list of sentences (`NLTK`)
    - dealing with punctuation and numbers (`Regular expressions`)
    - convert to lower case
    - split into individual words (called "tokenization" in `NLP` lingo)
- flatify list of reviews where each review is a list of sentences where each sentence is a list of words to list of sentences where each sentence is a list of words
- training (by created sentences) Word2Vec model and saving it
- reading the test and training data
- data cleaning and text preprocessing of test and training data
    - removing HTML markup (``BeautifulSoup``)
    - dealing with punctuation and numbers (``Regular expressions``)
    - convert to lower case
    - split into individual words (called "tokenization" in ``NLP`` lingo)
    - dealing with stopwords (``NLTK``)
- average-vectors: create word vectors from test and training data (using Word2Vec model)
- average-vectors: train classifiers using created in previous step, training word vectors
- average-vectors: make a predictions using created test word vectors
- average-vectors: calculate accuracy of predictions and save the results
- bag-of-centroids: create word vectors from test and training data (using Word2Vec model)
- bag-of-centroids: train classifiers using created in previous step, training word vectors
- bag-of-centroids: make a predictions using created test word vectors
- bag-of-centroids: calculate accuracy of predictions and save the results



# Files

`README.md` - project description in English
`aclImdb_v1.tar.gz` - dataset of reviews
`config.py` - project configuration (paths to dataset)
`parsers.py` - text parsers
`requirements.txt` - dependencies
`utils.py` - helpful and reusable functions
`bag_of_words/classifiers.py` - classifiers to make a predictions
`bag_of_words/run.py` - BagOfWords algorithm
`word_2_vec/run.py` - Word2Vec algorithm


# Results

Results of particular classifier for BagOfWords are stored in file:
``bag-of-words-*-model.csv``

Summary results for BagOfWords are stored in file:
``bag-of-words-summary.txt``

Results of particular classifier for Word2Vec average-vectors are stored in file:
``word-2-vec-average-vectors-*-model.csv``

Summary results for Word2Vec average-vectors are stored in file:
``word-2-vec-average-vectors-summary.txt``

Results of particular classifier for Word2Vec bag-of-centroids are stored in file:
``word-2-vec-bag-of-centroids-*-model.csv``

Summary results for Word2Vec bag-of-centroids are stored in file:
``word-2-vec-bag-of-centroids-summary.txt``


# Interpretation of results

As it's seen, BagOfWords is better then Word2Vec. The biggest reason is in averaging the vectors and using the centroids lose the order of words, making it very similar to the concept of BagOfWords. The fact that the performance is similar (within range of standard error) makes given two methods practically equivalent.

Training Word2Vec on a lot more text should greatly improve performance. Google's results are based on word vectors that were learned out of more than a billion-word. Additionaly, Word2Vec provides functions to load any pre-trained model that is output by Google's original *C tool*, so it's also possible to train a model in *C* and then import it into *Python*.


# Matrials

https://www.kaggle.com/c/word2vec-nlp-tutorial
