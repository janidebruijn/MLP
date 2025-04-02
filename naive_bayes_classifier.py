import time
import spacy
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn import metrics
from spacy.tokens import Doc
from sklearn.utils import class_weight
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer


def get_features(sentence_list):
    data = []
    for sent in tqdm(sentence_list, desc='Getting features'):
        data += [
            [
                word,
                len(word),  # word length
                1 if word.endswith('ing')
                else 0,  # ends with "ing"
                1 if word.endswith(('ito', 'ita'))
                else 0,  # ends with 'ito' or 'ita'
                1 if any(c in 'ñÑ' for c in word)
                else 0,  # contains ñ or Ñ
                1 if word.isupper()
                else 0,  # all uppercase
                1 if any(c.isdigit() for c in word)
                else 0,  # contains a number
                1 if any(c in '!@#$%^&*()-_+=<>?/{}[]' for c in word)
                else 0,  # contains special characters
                1 if word[0].isupper()
                else 0,  # starts with a capital letter
                1 if any(word[i] == word[i+1] for i in range(len(word)-1))
                else 0,  # contains repeated letters
                sum(1 for c in word.lower()
                    if c in 'aeiou'),  # vowel count
                sum(1 for c in word.lower()
                    if c.isalpha() and c not in 'aeiou'),  # consonant count
                1 if '-' in word
                else 0,  # contains hyphen
            ] for word in sent
        ]
    df = pd.DataFrame(
        data,
        columns=[
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
            'contains_hyphen',
            ]
        )
    return df


start = time.time()

train_sents = np.load('train_words.npy', allow_pickle=True)
train_labels_sents = np.load('train_labels.npy', allow_pickle=True)

n_sents = None
batch_size = 5000
model = MultinomialNB(alpha=0.1)
train_feats = get_features(train_sents[:n_sents])
train_labels = [label for sent in train_labels_sents[:n_sents]
                for label in sent]

vectoriser = CountVectorizer()
word_vectors = vectoriser.fit_transform(train_feats['word'])
train_feats = pd.concat([
    pd.DataFrame.sparse.from_spmatrix(
        word_vectors, columns=vectoriser.get_feature_names_out()),
    train_feats.drop(columns=["word"]),
], axis=1)

unique_classes = np.unique(train_labels)
class_weights = class_weight.compute_class_weight(
    'balanced', classes=unique_classes, y=train_labels)
class_weights_dict = {
    unique_classes[i]: class_weights[i]
    for i in range(len(unique_classes))
}

indexes = [[index[0], index[-1] + 1] for index
           in np.array_split(range(train_feats.shape[0] + 1),
                             len(train_feats) // batch_size)]
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

dev_sents = np.load('dev_words.npy', allow_pickle=True)
dev_labels_sents = np.load('dev_labels.npy', allow_pickle=True)

dev_feats = get_features(dev_sents[:n_sents])
dev_labels = [label for sent in dev_labels_sents[:n_sents] for label in sent]

doc_term_matrix = vectoriser.transform(dev_feats['word'])
dev_feats = pd.concat([
    pd.DataFrame.sparse.from_spmatrix(
        doc_term_matrix, columns=vectoriser.get_feature_names_out()),
    dev_feats.drop(columns=["word"])
], axis=1)

indexes = [[index[0], index[-1] + 1] for index
           in np.array_split(range(dev_feats.shape[0] + 1),
                             len(dev_feats) // batch_size)]

dev_predictions = [
    prediction for start, end in tqdm(indexes, desc='Predicting')
    for prediction in model.predict(dev_feats[start:end])
]

print(metrics.classification_report(
    dev_labels, dev_predictions, digits=3, zero_division=0))

time_taken = time.time() - start
print(f'Took {time_taken // 60} minutes and {time_taken % 60} seconds')
