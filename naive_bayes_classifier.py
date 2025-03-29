# code for naive bayes classifier from assignment 1, needs to be modified to work with our task


import numpy as np
from sklearn.naive_bayes import MultinomialNB
import abc
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer


class CustomClassifier(abc.ABC):
    def __init__(self):
        self.counter = None

    def get_features(self, text_list, ngram=1, max_features=5000):
        if self.counter is None:
            self.counter = CountVectorizer(ngram_range=(1, ngram), max_features=max_features)
            features_matrix = self.counter.fit_transform(text_list)
        else:
            features_matrix = self.counter.transform(text_list)

        return features_matrix.toarray()

    def tf_idf(self, text_feats):
        tfidf_transformer = TfidfTransformer().fit(text_feats)
        return tfidf_transformer.transform(text_feats)

    @abc.abstractmethod
    def fit(self, train_features, train_labels):
        """ Abstract method to be separately implemented for each custom classifier. """
        pass

    @abc.abstractmethod
    def predict(self, test_features):
        """ Abstract method to be separately implemented for each custom classifier. """
        pass


"""
Implement a Naive Bayes classifier with required functions:

fit(train_features, train_labels): to train the classifier
predict(test_features): to predict test labels 
"""


class CustomNaiveBayes(CustomClassifier):

    def __init__(self, alpha=1.0):
        """Initialize Naive Bayes classifier with smoothing factor alpha."""
        super().__init__()
        self.alpha = alpha
        self.prior = None
        self.classifier = MultinomialNB(alpha=self.alpha) 

    def fit(self, train_feats, train_labels):
        """Train the Naive Bayes classifier."""
        # of texts for that class / total # of texts
        unique_labels, counts = np.unique(train_labels, return_counts=True)
        self.prior = counts / len(train_labels)
        
        # Fit Naive Bayes model with training features and labels
        self.classifier.fit(train_feats, train_labels)
        
        # Mark model as trained 
        self.is_trained = True
        return self

    def predict(self, test_feats):
        """Predict class labels for test data."""
        assert self.is_trained, 'Model must be trained before predicting'
        
        # Use the trained model to predict labels for test data
        predictions = self.classifier.predict(test_feats)

        return predictions


