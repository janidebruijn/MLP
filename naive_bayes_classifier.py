import time
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn import metrics
from sklearn.utils import class_weight
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer


def get_features(sentence_list):
    '''
    Gets certain features from a list of lists like:
        ```
            list([
                list(['example', 'sentence']),
                list(['another', 'example', 'sentence']),
                ...
            ])
        ```
    Features are (in order):
        Token/word itself (turned into vectors later)
        Word length
        1/0 if word ends with "ing"
        1/0 if word ends with 'ito' or 'ita'
        1/0 if word contains ñ or Ñ
        1/0 if word is all uppercase
        1/0 if word contains a number
        1/0 if word contains special characters !@#$%^&*()-_+=<>?/{}[]
        1/0 if word starts with a capital letter
        1/0 if word contains 2 repeated letters
        Vowel count
        Consonant count
    Returns pandas dataframe containing the features with columns:
        'word',
        'word_length',
        'ends_with_ing',
        'ends_with_ito_ita',
        'contains_n_tilde',
        'is_all_uppercase',
        'contains_number',
        'contains_special_char',
        'start_with_capital',
        'contains_repeated_letters',
        'vowel_count',
        'consonant_count',
        'contains_hyphen'
    '''
    # initialise list
    data = []
    # for sentence in sentence list with tqdm progress bar
    for sent in tqdm(sentence_list, desc='Getting features'):
        # add features to list
        data += [
            [
                word,
                len(word),
                1 if word.endswith('ing') else 0,
                1 if word.endswith(('ito', 'ita')) else 0,
                1 if 'ñ' in word or 'Ñ' in word else 0,
                1 if word.isupper() else 0,
                1 if any(c.isdigit() for c in word) else 0,
                1 if any(c in '!@#$%^&*()-_+=<>?/{}[]' for c in word) else 0,
                1 if word[0].isupper() else 0,
                1 if any(word[i] == word[i+1] for i in range(len(word)-1))
                else 0,
                sum(1 for c in word.lower() if c in 'aeiou'),
                sum(1 for c in word.lower()
                    if c.isalpha() and c not in 'aeiou'),
                1 if '-' in word else 0,
            ] for word in sent
        ]
    # turn data into dataframe
    df = pd.DataFrame(
        data,
        columns=[
            'word', 'word_length', 'ends_with_ing', 'ends_with_ito_ita',
            'contains_n_tilde', 'is_all_uppercase', 'contains_number',
            'contains_special_char', 'start_with_capital',
            'contains_repeated_letters', 'vowel_count', 'consonant_count',
            'contains_hyphen',
        ]
    )
    return df


# start timer
start = time.time()

# load train sentences and labels
train_sents = np.load('train_words.npy', allow_pickle=True)
train_labels_sents = np.load('train_labels.npy', allow_pickle=True)

# intialise model
model = MultinomialNB(alpha=0.1)

# size of subset in number of sentences (None for all)
n_sents = None

# get train features
train_feats = get_features(train_sents[:n_sents])
# unpack label sentences to individual labels
train_labels = [label for sent in train_labels_sents[:n_sents]
                for label in sent]

# intialise count vectoriser
vectoriser = CountVectorizer()
# fit vectoriser to train words and transform train words to vectors
train_word_vectors = vectoriser.fit_transform(train_feats['word'])
# add train word vectors to train dataframe
train_feats = pd.concat([
    pd.DataFrame.sparse.from_spmatrix(
        train_word_vectors, columns=vectoriser.get_feature_names_out()),
    train_feats.drop(columns=["word"]),
], axis=1)

# get unique classes
unique_classes = np.unique(train_labels)
# get unqiue class weights
class_weights = class_weight.compute_class_weight(
    'balanced', classes=unique_classes, y=train_labels)
# construct dict for easier access
class_weights_dict = {
    unique_classes[i]: class_weights[i]
    for i in range(len(unique_classes))
}

# set batch size
# setting too high generates large dense arrays
batch_size = 5000
# get batch start/end indexes
indexes = [[index[0], index[-1] + 1] for index
           in np.array_split(range(train_feats.shape[0] + 1),
                             len(train_feats) // batch_size)]
# partial fit to prevent too large dense arrays being made
for start, end in tqdm(indexes, desc='Fitting'):
    model.partial_fit(
        train_feats[start:end],
        train_labels[start:end],
        classes=unique_classes,
        sample_weight=[
            class_weights_dict[label]
            for label in train_labels[start:end]
        ]
    )

# load dev sentences and labels
dev_sents = np.load('dev_words.npy', allow_pickle=True)
dev_labels_sents = np.load('dev_labels.npy', allow_pickle=True)

# get dev features
dev_feats = get_features(dev_sents[:n_sents])
# unpack label sentences to individual labels
dev_labels = [label for sent in dev_labels_sents[:n_sents] for label in sent]

# transform dev words to vectors
dev_word_vectors = vectoriser.transform(dev_feats['word'])
# add dev word vectors to dev dataframe
dev_feats = pd.concat([
    pd.DataFrame.sparse.from_spmatrix(
        dev_word_vectors, columns=vectoriser.get_feature_names_out()),
    dev_feats.drop(columns=["word"])
], axis=1)

# get batch start/end indexes
indexes = [[index[0], index[-1] + 1] for index
           in np.array_split(range(dev_feats.shape[0] + 1),
                             len(dev_feats) // batch_size)]

# predict dev words
dev_predictions = [
    prediction for start, end in tqdm(indexes, desc='Predicting')
    for prediction in model.predict(dev_feats[start:end])
]

# print classification report
print(metrics.classification_report(
    dev_labels, dev_predictions, digits=3, zero_division=0))

# print time taken
time_taken = time.time() - start
print(f'Took {time_taken // 60} minutes and {time_taken % 60} seconds')
