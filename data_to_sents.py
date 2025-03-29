import os
import numpy as np


def data_to_sents(target_paths):
    '''
    Takes .conll files from '/lid_spaeng' with layout
        ```
            # sent_enum = 1
            example    lang1
            sentence    lang1

            # sent_enum = 2
            another    lang1
            example    lang1
            sentence    lang1
            ...
        ```
     and converts them to .npy files with structure
        ```
            np.array([
                list(['example', 'sentence']),
                list(['another', 'example', 'sentence'])
            ])
        ```
    Load the .npy files with `np.load([.npy], allow_pickle=True)`
    '''
    # check for invalid files
    if not all(target_path[-6:] == '.conll' for target_path in target_paths):
        raise ValueError('Non .conll file in target paths')

    # check for data
    if 'lid_spaeng' not in os.listdir():
        raise FileNotFoundError('lid_spaeng not found')

    # get file paths in lid_spaeng
    data_dir_path = os.getcwd() + '/lid_spaeng/'
    data_file_paths = os.listdir(data_dir_path)

    # get target existing target paths
    existing_target_paths = [
        target_path for target_path in target_paths
        if target_path in data_file_paths
    ]

    # check for existing target paths
    if existing_target_paths == []:
        raise ValueError('No valid targets found')

    # get sents
    for data_file_path in data_file_paths:
        # intialise lists
        words = []
        labels = []
        # open file
        with open(data_dir_path + data_file_path) as f:
            # initialise nested lists
            cur_words = [[]]
            cur_labels = [[]]
            # loop through lines
            for line in f:
                # split
                splitted = line.split()
                # if line like 'word    label' (train/dev.conll)
                if len(splitted) == 2:
                    # add to respective lists
                    cur_words[-1].append(splitted[0])
                    cur_labels[-1].append(splitted[1])
                # if line like 'word' (test.conll)
                if len(splitted) == 1:
                    cur_words[-1].append(splitted[0])
                # if line is empty
                if splitted == []:
                    # insert empty list for next sentence
                    cur_words.append([])
                    cur_labels.append([])

        # when done, add words/labels to whole
        words += cur_words[:-1]
        labels += cur_labels[:-1]

        # save respective lists
        np.save(f'{data_file_path.split(".")[0]}_words.npy',
                np.array(words, dtype=object))
        # dont save if empty
        if not all([sent == [] for sent in labels]):
            np.save(f'{data_file_path.split(".")[0]}_labels.npy',
                    np.array(labels, dtype=object))


if __name__ == '__main__':
    data_to_sents(['dev.conll', 'train.conll'])
