import os
import warnings
import numpy as np

with warnings.catch_warnings():  
    warnings.filterwarnings("ignore",category=FutureWarning)
    import tensorflow as tf

    from C10_resources import FullyconnectedClassifier

from datetime import datetime

# read dataset
(train_data, train_target), (test_data, test_target) = tf.keras.datasets.mnist.load_data()
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

train_spec = tf.estimator.TrainSpec(
    input_fn=tf.compat.v1.estimator.inputs.numpy_input_fn(x = {'f1' : X_train},
                                   y = y_train,
                                   batch_size=batch_size,
                                   num_epochs=None,
                                   shuffle=True), 
    max_steps=max_steps)                        

eval_spec = tf.estimator.EvalSpec(
    input_fn=tf.compat.v1.estimator.inputs.numpy_input_fn(x = {'f1' : X_test},y = y_test,
                                   num_epochs=None,
                                   shuffle=False))

tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)