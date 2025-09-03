# This script takes the datasets created by the authors 
# and creates more balanced datasets.

import numpy as np
import pickle

data_path = "/mnt/POD/NNDL_gd/GW_data/data" 
dataset_path = "/mnt/POD/NNDL_gd/GW_data/datasets" 

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
    print('X shape', X.shape)

    # standardize
    print(label)
    mean = X.mean(axis=1, keepdims=True)
    print('mean shape', mean.shape)
    std = X.std(axis=1, keepdims=True)
    print('std shape', std.shape)
    X = (X-mean)/std
    print('std X shape', X.shape)

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
    signal_dataset = load_concat('signal')
    glitch_dataset = load_concat('glitch')

    # number of samples
    N = len(noise_dataset)
    S = len(signal_dataset)
    G = len(glitch_dataset)
    N_samples = N+S+G

    print('Total samples: ', N_samples)

    # split datasets
    noise_train, noise_valid, noise_test = dataset_split(noise_dataset)
    signal_train, signal_valid, signal_test = dataset_split(signal_dataset)
    glitch_train, glitch_valid, glitch_test = dataset_split(glitch_dataset)

    # merge by labels
    training_dataset = np.concatenate((noise_train, signal_train, glitch_train))
    validation_dataset = np.concatenate((noise_valid, signal_valid, glitch_valid))
    test_dataset = np.concatenate((noise_test, signal_test, glitch_test))

    # shuffle
    np.random.shuffle(training_dataset)
    np.random.shuffle(validation_dataset)
    np.random.shuffle(test_dataset)

    # save datasets
    training_file = dataset_path+'/training_data.pkl'
    with open(training_file, 'wb') as file:
        pickle.dump(training_dataset, file)

    validation_file = dataset_path+'/validation_data.pkl'
    with open(validation_file, 'wb') as file:
        pickle.dump(validation_dataset, file)

    test_file = dataset_path+'/test_data.pkl'
    with open(test_file, 'wb') as file:
        pickle.dump(test_dataset, file)