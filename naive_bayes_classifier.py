import time
import spacy
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn import metrics
from spacy.tokens import Doc
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer


def get_features(sentence_list):
    data = []
    for sent in tqdm(sentence_list, desc='Getting features'):
        doc_en = Doc(nlp_en.vocab, words=sent)
        doc_sp = Doc(nlp_sp.vocab, words=sent)

        doc_en = nlp_en(doc_en)
        doc_sp = nlp_sp(doc_sp)

        data += [
            [
                token_en.text,
                hash(token_en.pos_) % 1000,
                hash(token_sp.pos_) % 1000,
                1 if ('ñ' in token_en.text or 'Ñ' in token_en.text) else 0,
                1 if token_sp.text.endswith(('ito', 'ita')) else 0,
            ] for token_en, token_sp in zip(doc_en, doc_sp)
        ]
    df = pd.DataFrame(
        data,
        columns=['word', 'en_pos', 'sp_pos', 'has_ñÑ', 'diminutive'])
    return df


def mins(seconds):
    return f'{seconds // 60} mins and {seconds % 60} secs'


start = time.perf_counter()

nlp_en = spacy.load(
    'en_core_web_sm', disable=['parser', 'attribute_ruler', 'lemmatizer'])
nlp_sp = spacy.load(
    'es_core_news_sm', disable=['parser', 'attribute_ruler', 'lemmatizer'])

train_sents = np.load('train_words.npy', allow_pickle=True)
train_labels_sents = np.load('train_labels.npy', allow_pickle=True)

amount = None
model = MultinomialNB()
train_feats = get_features(train_sents[:amount])
train_labels = [label for sent in train_labels_sents[:amount]
                for label in sent]

vectoriser = CountVectorizer()
doc_term_matrix = vectoriser.fit_transform(train_feats['word'])
train_feats = pd.concat([
    pd.DataFrame.sparse.from_spmatrix(
        doc_term_matrix, columns=vectoriser.get_feature_names_out()),
    train_feats.drop(columns=["word"])
], axis=1)

indexes = [[index[0], index[-1] + 1] for index
           in np.array_split(range(train_feats.shape[0] + 1),
                             len(train_feats) // 2500)]
for start, end in tqdm(indexes, desc='Fitting'):
    model.partial_fit(train_feats[start:end],
                      train_labels[start:end],
                      classes=np.unique(train_labels))

dev_sents = np.load('dev_words.npy', allow_pickle=True)
dev_labels_sents = np.load('dev_labels.npy', allow_pickle=True)

dev_feats = get_features(dev_sents[:amount])
dev_labels = [label for sent in dev_labels_sents[:amount] for label in sent]

doc_term_matrix = vectoriser.transform(dev_feats['word'])
dev_feats = pd.concat([
    pd.DataFrame.sparse.from_spmatrix(
        doc_term_matrix, columns=vectoriser.get_feature_names_out()),
    dev_feats.drop(columns=["word"])
], axis=1)

indexes = [[index[0], index[-1] + 1] for index
           in np.array_split(range(dev_feats.shape[0] + 1),
                             len(dev_feats) // 2500)]

dev_predictions = [
    prediction for start, end in tqdm(indexes, desc='Predicting')
    for prediction in model.predict(dev_feats[start:end])
]

print(metrics.classification_report(
    dev_labels, dev_predictions, digits=3, zero_division=0))

time_taken = start - time.perf_counter()
print(f'Took {mins(time_taken)}')
