import os
import warnings
import numpy as np
with warnings.catch_warnings():  
    warnings.filterwarnings("ignore",category=FutureWarning)
    import tensorflow as tf 
    from C10_resources import FullyconnectedClassifier

from datetime import datetime

from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler




# read dataset
dataset = fetch_openml('mnist_784', version=1)
n_classes = len(set(dataset.target))
train_data, test_data, train_target, test_target = train_test_split(dataset.data, dataset.target)

# standardization trained on train, reused on test dataset
scaler = StandardScaler()
X_train = scaler.fit_transform(train_data)
X_test = scaler.transform(test_data)

# string to class index
class_index_map = { k: v for v, k in enumerate(sorted(list(set(dataset.target))))}
index_class_map = { v: k for k,v in class_index_map.items() }
class_index_mapping = np.vectorize(lambda x: class_index_map[x])

y_train = class_index_mapping(train_target).reshape(-1,1)
y_test = class_index_mapping(test_target).reshape(-1,1)

# create feature columns
feature_columns = [ tf.feature_column.numeric_column(key='f1', shape=X_train[0].shape) ]

batch_size=50
max_steps=40000
save_checkpoints_steps=2000
log_step_count_steps=500

# this seed is used only for initialization
# batch is still random with no chance to set the seed
# see: https://stackoverflow.com/questions/47009560/tf-estimator-shuffle-random-seed
config = tf.estimator.RunConfig(tf_random_seed=42,
                                model_dir=os.path.join('tmp',
                                                       datetime.utcnow().strftime('%Y%m%d-%H%M%S')),
                                save_checkpoints_steps=save_checkpoints_steps,
                                log_step_count_steps=log_step_count_steps)

estimator = tf.estimator.Estimator(
    model_fn=FullyconnectedClassifier.model_fn,
    params={'feature_columns': feature_columns,
        'hidden_units': [300, 100],
        'activation': 'relu',
        'n_classes': 10,
        'learning_rate': 1e-4
       },
    config=config
    )

train_spec = tf.estimator.TrainSpec(input_fn=tf.estimator.inputs.numpy_input_fn(x = {'f1' : X_train},
                                                                                y = y_train, 
                                                                                batch_size=batch_size, 
                                                                                num_epochs=None,
                                                                                shuffle=True), 
                                    max_steps=max_steps)                        

eval_spec = tf.estimator.EvalSpec(input_fn=tf.estimator.inputs.numpy_input_fn(x = {'f1' : X_test},
                                                                              y = y_test, 
                                                                              num_epochs=None,
                                                                              shuffle=False))

tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)