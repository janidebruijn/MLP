import os
import ast
import numpy as np
from pathlib import Path
from sklearn import metrics


def get_dict_baseline(sents_path, true_label_sents_path):
    '''
    Returns sklear.metrics.classification_report for sents_path and
     true_label_sents_path based on word list sets found in /wordlists
    Takes paths to numpy arrays containing lists of words
    '''
    # check for sents and label files
    if not all([path in os.listdir(os.getcwd())
                for path in [sents_path, true_label_sents_path]]):
        raise FileNotFoundError('Words or labels not found')

    # load sents and labels
    train_sents = np.load(sents_path, allow_pickle=True)
    true_labels = np.load(true_label_sents_path, allow_pickle=True)

    # check for wordlists directory
    if 'wordlists' not in os.listdir():
        raise FileNotFoundError('wordslists directory not found')

    # get wordlists/ file paths
    data_dir_path = os.getcwd() + '/wordlists/'
    lang_set_paths = [path for path in os.listdir(data_dir_path)
                      if path[:4] == 'set_' and path[-4:] == '.txt']

    # check amount of set files
    if len(lang_set_paths) != 3:
        raise ValueError(f'Not enough language sets found, only found \
                         {len(lang_set_paths)}')

    # load sets into dict
    lang_dicts = {
        path[4:-4]: ast.literal_eval(Path(data_dir_path + path).read_text())
        for path in lang_set_paths
    }

    # convert sents and label sents into singular lists
    train_words = [word for sentence in train_sents for word in sentence]
    true_labels = [label for sentence in true_labels for label in sentence]

    # check both lists are the same length
    if len(train_words) != len(true_labels):
        raise ValueError(f'Length of words not the same as labels ( \
                         {len(train_words)}, {len(true_labels)})')

    # assign predictions based on dicts
    predicted_labels = []
    for word in train_words:
        for key in lang_dicts.keys():
            if word in lang_dicts[key]:
                predicted_labels.append(key)
                break
        else:
            predicted_labels.append('other')

    # return classification report
    return metrics.classification_report(
        true_labels, predicted_labels, digits=3, zero_division=0)


if __name__ == '__main__':
    print(get_dict_baseline('test_train_words.npy', 'test_train_labels.npy'))
