import os

import numpy as np
import pandas as pd

def load_training_and_test_data(filename):
    with open(os.path.join("..", "data", filename), "rb") as f:
            data = pd.read_csv(f,header=0)

    X = data.values[:,:-1]
    y = data.values[:,-1:]

    # Handling for specific files

    if filename == "breast-cancer-wisconsin-data.csv":
        data['diagnosis'] = data['diagnosis'].map({'M': 1, 'B': 0})
        
        X = data.values[:,2:-1]
        y = data['diagnosis'].values

    # Shuffle the dataset

    shuffled_indices = np.arange(X.shape[0])
    np.random.shuffle(shuffled_indices)

    X = X[shuffled_indices]
    y = y[shuffled_indices]

    # Split the dataset into training and test data

    split_index = int(np.ceil(X.shape[0]*0.8))
   
    Xtest = X[split_index:,:]
    ytest = y[split_index:] 
    X = X[:split_index,:]
    y = y[:split_index]

    return X, y, Xtest, ytest