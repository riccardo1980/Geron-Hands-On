import os
import warnings
import numpy as np

with warnings.catch_warnings():  
    warnings.filterwarnings("ignore",category=FutureWarning)
    import tensorflow as tf

    #from C11_resources import FullyconnectedClassifier

#from datetime import datetime

def dataset_prune(features, labels, idx):
    return features[idx], labels[idx]

# read complete dataset
(complete_train_data, complete_train_target), (complete_test_data, complete_test_target) = tf.keras.datasets.mnist.load_data()

# retain digits 0 to 4
train_data, train_target = dataset_prune(complete_train_data, 
                                         complete_train_target,
                                         (complete_train_target < 5))

test_data, test_target = dataset_prune(complete_test_data,
                                       complete_test_target,
                                       (complete_test_target < 5))

# cast features to float32, targets to int32
train_data = train_data.astype(np.float32)
test_data = test_data.astype(np.float32) 
train_target = train_target.astype(np.int32)
test_target = test_target.astype(np.int32) 

n_classes = len(set(train_target))

print('n_classes: {}'.format(n_classes))

# standardization trained on train, reused on test dataset
def preproc_standardize(dataset, mean, std):
    return (dataset - mean) / (std + 1e-15)

features_mean = np.mean(train_data, axis=0)
features_std = np.std(train_data, axis=0)

X_train = preproc_standardize(train_data, features_mean, features_std)
X_test =  preproc_standardize(test_data, features_mean, features_std)

y_train = train_target
y_test = test_target

# create feature columns
feature_columns = [ tf.feature_column.numeric_column(key='f1', shape=X_train[0].shape) ]

n_classes = len(set(train_target))
learning_rate = 1e-4
params={'feature_columns': feature_columns,
    'hidden_units': 5*[100],
    'activation': 'elu',
    'n_classes': n_classes,
    'optimizer': tf.compat.v1.train.AdagradOptimizer(learning_rate=learning_rate),
    'batch_norm_momentum': None
}