# This script merges noise and glitch into one class ("background"=0)
# and keeps signal as the other class ("signal"=1).

import numpy as np
import pickle

data_path = "/mnt/POD/NNDL_gd/GW_data/data" 

# dataset splitting percentages
p_train = 0.6
p_valid = 0.15
p_test = 0.25

seed = 0


def load_concat(label, path=data_path):
    """
    Load files and create a dataset for each class.
    """
    f = np.load(path+'/'+label+'_train.npz')
    X_train, Y_train = f['X'], f['Y']

    f = np.load(path+'/'+label+'_test.npz')
    X_test, Y_test = f['X'], f['Y']

    X = np.concatenate((X_train, X_test), axis=0)
    Y = np.concatenate((Y_train, Y_test), axis=0)
    Y = Y.reshape(-1,1)

    data = np.hstack((X,Y))
    return data


def dataset_split(dataset, train=p_train, valid=p_valid, test=p_test):
    """
    Split each dataset into train, validation and test.
    """
    N = len(dataset)
    n_train = int(p_train*N)
    n_valid = int(p_valid*N)
    n_test = int(p_test*N)

    np.random.shuffle(dataset) 

    train = dataset[:n_train]
    valid = dataset[n_train:n_train+n_valid]
    test = dataset[-n_test:]

    return train, valid, test


# --- MAIN --- #
if __name__=="__main__":

    np.random.seed(seed)

    # load data
    noise_dataset = load_concat('noise')
    glitch_dataset = load_concat('glitch')
    signal_dataset = load_concat('signal')

    # relabel: background=0, signal=1
    noise_dataset[:,-1] = 0
    glitch_dataset[:,-1] = 0
    signal_dataset[:,-1] = 1

    # merge noise and glitch into background
    background_dataset = np.concatenate((noise_dataset, glitch_dataset), axis=0)

    # number of samples
    B = len(background_dataset)
    S = len(signal_dataset)
    N_samples = B + S

    print('Total samples: ', N_samples)
    print('Background samples:', B)
    print('Signal samples:', S)

    # split datasets
    background_train, background_valid, background_test = dataset_split(background_dataset)
    signal_train, signal_valid, signal_test = dataset_split(signal_dataset)

    # merge by labels
    training_dataset = np.concatenate((background_train, signal_train))
    validation_dataset = np.concatenate((background_valid, signal_valid))
    test_dataset = np.concatenate((background_test, signal_test))

    # shuffle
    np.random.shuffle(training_dataset)
    np.random.shuffle(validation_dataset)
    np.random.shuffle(test_dataset)

    # save datasets
    training_file = 'training_data_01.pkl'
    with open(training_file, 'wb') as file:
        pickle.dump(training_dataset, file)

    validation_file = 'validation_data_01.pkl'
    with open(validation_file, 'wb') as file:
        pickle.dump(validation_dataset, file)

    test_file = 'test_data_01.pkl'
    with open(test_file, 'wb') as file:
        pickle.dump(test_dataset, file)