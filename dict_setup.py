import os


def construct_set_files():
    '''
    Takes raw text files containing a word per line from /wordlists, converts
     them to sets, makes set containing overlapping words and removes them
     from both sets, and writes them to `/wordlists/set_[filename].txt`
    To use the sets, initialise a set like:
        ```
            import ast
            with open('wordlists/set_english.txt') as f:
                english_word_set = ast.literal_eval(f.read())
        ```
     or for single line initialisation, like in 'dict_baseline.py':
        ```
            ast.literal_eval(Path('wordlists/set_english.txt').read_text())
        ```
    '''

    cwd = os.getcwd()
    # check for wordlists directory
    if 'wordlists' not in os.listdir(cwd):
        raise FileNotFoundError('/wordlists was not found')
    # get path to read from and write to
    data_dir_path = f'{cwd}/wordlists/'
    # get file paths to read from
    data_file_paths = os.listdir(data_dir_path)
    # check for empty path list
    if not data_file_paths:
        raise FileNotFoundError('No files found')

    # initialise dict for storing and making sets
    word_sets = dict()
    for data_file_path in data_file_paths:
        # skip files we do not read from
        if data_file_path[-4:] != '.txt' or data_file_path[:3] == 'set':
            continue
        # open and add set of words to dict
        with open(data_dir_path + data_file_path, 'r') as file:
            word_sets[data_file_path.split('.')[0]] = {
                line.rstrip() for line in file
            }

    # check for empty sets
    if any([value == {} for _, value in word_sets.items()]):
        raise ValueError('1 or more word set is empty')
    # get keys
    keys = list(word_sets.keys())
    # check if two word lists are used
    if len(keys) != 2:
        raise ValueError('More than 2 word lists were found')

    # make overlap set and remove from the others
    word_sets['ambiguous'] = word_sets[keys[0]] & word_sets[keys[1]]
    # check for empty overlap set
    if word_sets['ambiguous'] == set():
        if 'ambiguous' in word_sets:
            del word_sets['ambiguous']
        print('No overlap found between word lists')
    else:
        word_sets[keys[0]] -= word_sets['ambiguous']
        word_sets[keys[1]] -= word_sets['ambiguous']

    # write to respective files
    for key in word_sets.keys():
        with open(data_dir_path + f'set_{key}.txt', 'w') as file:
            file.write(str(word_sets[key]))


if __name__ == '__main__':
    construct_set_files()
