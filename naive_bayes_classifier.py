# code for naive bayes classifier from assignment 1, needs to be modified to work with our task
# zit gwn ff wat random code in wat miss handig is

import nltk
import spacy
import numpy as np
from sklearn.naive_bayes import MultinomialNB
import abc
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from nltk.tokenize import word_tokenize
from nltk import pos_tag

# Load Spacy model for spanish and english
nlp = spacy.load("en_core_web_sm")  # or "es_core_news_sm" for Spanish



class CustomClassifier(abc.ABC):
    def __init__(self):
        self.counter = None

    def get_features(self, text_list, ngram=1, max_features=5000):
        """Extracts features including:
        - word-level n-grams
        - pos tags
        - character-based
        """
        # n-grams using CountVectorizer
        if self.counter is None:
            self.counter = CountVectorizer(ngram_range=(1, ngram), max_features=max_features)
            features_matrix = self.counter.fit_transform(text_list)
        else:
            features_matrix = self.counter.transform(text_list)

        features_matrix = features_matrix.toarray()

        # POS tagging and character features
        pos_tags = []
        char_features = []

        for sentence in text_list:
            doc = nlp(sentence)
            sentence_pos_tags = []
            sentence_char_features = []

            # Align spaCy tokens with CountVectorizer tokens
            for token in doc:
                if not token.is_punct and not token.is_space:
                    sentence_pos_tags.append(hash(token.pos_) % 1000)
                    has_ñ = 1 if 'ñ' in token.text else 0
                    has_diminutive = 1 if token.text.endswith('ito') or token.text.endswith('ita') else 0
                    sentence_char_features.append([has_ñ, has_diminutive])

            # Append sentence-level features
            pos_tags.extend(sentence_pos_tags)
            char_features.extend(sentence_char_features)

        # Ensure the number of rows matches
        pos_features = np.array(pos_tags).reshape(-1, 1)
        char_features = np.array(char_features)

        # Adjust dimensions to match `features_matrix`
        if pos_features.shape[0] > features_matrix.shape[0]:
            print(f"Warning: POS features have more rows ({pos_features.shape[0]}) than features_matrix ({features_matrix.shape[0]}). Truncating.")
            pos_features = pos_features[:features_matrix.shape[0]]
        elif pos_features.shape[0] < features_matrix.shape[0]:
            print(f"Warning: POS features mismatch. Expected {features_matrix.shape[0]}, got {pos_features.shape[0]}. Padding.")
            pos_features = np.pad(pos_features, ((0, features_matrix.shape[0] - pos_features.shape[0]), (0, 0)), mode='constant')

        if char_features.shape[0] > features_matrix.shape[0]:
            print(f"Warning: Char features have more rows ({char_features.shape[0]}) than features_matrix ({features_matrix.shape[0]}). Truncating.")
            char_features = char_features[:features_matrix.shape[0]]
        elif char_features.shape[0] < features_matrix.shape[0]:
            print(f"Warning: Char features mismatch. Expected {features_matrix.shape[0]}, got {char_features.shape[0]}. Padding.")
            char_features = np.pad(char_features, ((0, features_matrix.shape[0] - char_features.shape[0]), (0, 0)), mode='constant')

        # Combine all features
        all_features = np.hstack((features_matrix, pos_features, char_features))
        return all_features

            

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
        self.is_trained = False  # Initialize is_trained to False

    def fit(self, train_feats, train_labels):
        """Train the Naive Bayes classifier."""
        # of texts for that class / total # of texts
        self.classifier.fit(train_feats, train_labels)  
        self.is_trained = True  
        return self

    def predict(self, test_feats):
        """Predict class labels for test data."""
        assert self.is_trained, 'Model must be trained before predicting'
    
        return self.classifier.predict(test_feats)




""" Testing code below, eventually needs to be in the main doc but this is just for running in this file"""

# Example usage of the CustomNaiveBayes class
# Example dataset: list of (word, label) pairs
train_data = [
    # English words (lang1)
    ("night", "lang1"), ("friends", "lang1"), ("sun", "lang1"), ("rain", "lang1"),
    
    # Spanish words (lang2)
    ("mañana", "lang2"), ("porque", "lang2"), ("libro", "lang2"), ("mundo", "lang2"),
    
    # Mixed language (mixed) - combination of English and Spanish
    ("#NumerosEverywhere", "mixed"), ("hello, mundo", "mixed"), ("#LoveLaVida", "mixed"),
    
    # Non-words (other) - punctuation, numbers
    ("11:11", "other"), (",", "other"), ("!", "other"), ("$", "other"), ("12345", "other"),
    
    # Unknown words (unk) - words not English, Spanish, or any recognized language
    ("boo", "unk"), ("pfftt", "unk"), ("xyzxyz", "unk"), ("blarg", "unk"),
    
    # Named entities (ne) - proper names, places, etc.
    ("Betty", "ne"), ("marialejandra", "ne"), ("Paris", "ne"), ("Einstein", "ne")
]

test_data = [
    # English words (lang1)
    "night", "friends", "sun", "rain",
    
    # Spanish words (lang2)
    "mañana", "porque", "libro", "mundo",
    
    # Mixed language (mixed)
    "#NumerosEverywhere", "hello, mundo", "#LoveLaVida",
    
    # Non-words (other)
    "11:11", ",", "!", "$", "12345",
    
    # Unknown words (unk)
    "boo", "pfftt", "xyzxyz", "blarg",
    
    # Named entities (ne)
    "Betty", "marialejandra", "Paris", "Einstein"
]


model = CustomNaiveBayes()

# Prepare training data
train_texts = [word for word, label in train_data]  # Extract words
train_labels = [label for word, label in train_data]  # Extract labels

# Extract features from training data
train_feats = model.get_features(train_texts)

# Train the model
model.fit(train_feats, train_labels)

train_predictions = model.predict(train_feats)

# Show results for the training data
print("\nPredictions on training data:")
for text, label, pred in zip(train_texts, train_labels, train_predictions):
    print(f"Text: {text} → True Label: {label}, Predicted Label: {pred}")

new_data = [
    "hello world",  # mixed language (if it's a combination of English/Spanish)
    "buenos días",  # Spanish
    "goodbye",      # English
    "12345",        # other (numbers)
    "!",            # other (punctuation)
    "newYork",      # named entity (ne)
    "¡cuidado!",    # Spanish
]

# Extract features from new data
new_data_feats = model.get_features(new_data)

# Make predictions on the new, unlabeled data
new_predictions = model.predict(new_data_feats)

# Show results for the new unlabeled data
print("\nPredictions on new unlabeled data:")
for text, pred in zip(new_data, new_predictions):
    print(f"Text: {text} → Predicted Language: {pred}")




def read_dataset(subset, split): # deze heb je niet echt nodig, maar heb m ff laten staan agz die functie hieronder m ook gebruikt
    print('***** Reading the dataset *****')

    fname = os.path.join("data", f'{subset}_{split}.csv')
    inputdf = pd.read_csv(fname, sep=",", encoding='utf-8', header=0)

    texts = inputdf["text_"].to_list()
    labels = inputdf["label"].to_list()

    assert len(texts) == len(labels), 'Error: there should be equal number of texts and labels.'
    print(f'Number of samples: {len(texts)}')

    return texts, labels

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
    train_feats = cls.tf_idf(train_feats)
    test_feats = cls.get_features(test_data)
    test_feats = cls.tf_idf(test_feats)

    cls.fit(train_feats, train_labels)

    # Predict labels for test data by using trained classifier and features of the test data
    predicted_test_labels = cls.predict(test_feats)

    # Evaluate the classifier by comparing predicted test labels and true test labels
    evaluate(test_labels, predicted_test_labels)
    
    return cls
