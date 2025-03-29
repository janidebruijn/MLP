# code for naive bayes classifier from assignment 1, needs to be modified to work with our task
# zit gwn ff wat random code in wat miss handig is


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



def train_test(classifier='svm'):
    # Read train and test data and generate tweet list together with label list
    
    # Choose the data set you plan to work with. Your options:
    # 'Books_5', 'Clothing_Shoes_and_Jewelry_5', 'Electronics_5'
    # 'Home_and_Kitchen_5', 'Kindle_Store_5', 'Movies_and_TV_5'
    # 'Pet_Supplies_5', 'Sports_and_Outdoors_5',
    # 'Tools_and_Home_Improvement_5', 'Toys_and_Games_5'
    subset = 'Books_5'
    train_data, train_labels = read_dataset(subset, 'train')
    test_data, test_labels = read_dataset(subset, 'test')

    # Preprocess train and test data
    train_data = preprocess_dataset(train_data)
    test_data = preprocess_dataset(test_data)

    # Create your custom classifier
    if classifier == 'svm':
        cls = SVMClassifier(kernel='linear')
    elif classifier == 'naive_bayes':
        cls = CustomNaiveBayes()
    # elif classifier == 'knn':
    #     cls = CustomKNN(k=5, distance_metric='cosine')

    # Generate features from train and test data
    # features: word count features per sentences as a 2D numpy array
    train_feats = cls.get_features(train_data)
    train_feats = cls.tf_idf(train_feats) # also using the tf-idf was an attempt to improve accuracy, and it improved the accuracy for both models
    test_feats = cls.get_features(test_data)
    test_feats = cls.tf_idf(test_feats) # also using the tf-idf was an attempt to improve accuracy, and it improved the accuracy for both models

    cls.fit(train_feats, train_labels)

    # Predict labels for test data by using trained classifier and features of the test data
    predicted_test_labels = cls.predict(test_feats)

    # Evaluate the classifier by comparing predicted test labels and true test labels
    evaluate(test_labels, predicted_test_labels)
    
    return cls
