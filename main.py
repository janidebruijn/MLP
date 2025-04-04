from data_to_sents import data_to_sents
from dict_setup import construct_set_files
from dict_baseline import get_dict_baseline
from naive_bayes_classifier import prediction

def classifier():
    data_to_sents(['dev.conll', 'train.conll']) # preprocess data
    construct_set_files() # construct wordlists
    report = get_dict_baseline('dev_words.npy', 'dev_labels.npy')  # get dict baseline
    print(f'Baseline report:\n{report}') 
    prediction() # naive bayes classifier
    
    
if __name__ == '__main__':
    classifier()

