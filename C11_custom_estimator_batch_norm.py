from datetime import datetime
import os
import warnings
import numpy as np

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=FutureWarning)
    import tensorflow as tf

    from C11_resources import FullyconnectedClassifier


# read dataset
(train_data, train_target), (test_data,
                             test_target) = tf.keras.datasets.mnist.load_data()
train_data = train_data.astype(np.float32)
test_data = test_data.astype(np.float32)
train_target = train_target.astype(np.int32)
test_target = test_target.astype(np.int32)

n_classes = len(set(train_target))

# standardization trained on train, reused on test dataset


def preproc_standardize(dataset, mean, std):
    return (dataset - mean) / (std + 1e-15)


features_mean = np.mean(train_data, axis=0)
features_std = np.std(train_data, axis=0)

X_train = preproc_standardize(train_data, features_mean, features_std)
X_test = preproc_standardize(test_data, features_mean, features_std)

y_train = train_target
y_test = test_target

# create feature columns
feature_columns = [tf.feature_column.numeric_column(
    key='f1', shape=X_train[0].shape)]

batch_size = 200
max_steps = 30000
save_checkpoints_steps = 2000  # checkpoints
log_step_count_steps = 100    # summary
# do not re-evaluate unless last eval is lolder than...
throttle_secs = 10

print('n_classes: {}'.format(n_classes))
print('epochs: {}'.format(max_steps * batch_size / y_train.shape[0]))
# this seed is used only for initialization
# batch is still random with no chance to set the seed
# see: https://stackoverflow.com/questions/47009560/tf-estimator-shuffle-random-seed
config = tf.estimator.RunConfig(tf_random_seed=42,
                                model_dir=os.path.join('tmp',
                                                       datetime.utcnow().strftime('%Y%m%d-%H%M%S')),
                                save_checkpoints_steps=save_checkpoints_steps,
                                log_step_count_steps=log_step_count_steps)

learning_rate = 1e-4
estimator = tf.estimator.Estimator(
    model_fn=FullyconnectedClassifier.model_fn,
    params={'feature_columns': feature_columns,
            'hidden_units': [300, 100],
            'activation': 'relu',
            'n_classes': n_classes,
            'optimizer': tf.compat.v1.train.AdagradOptimizer(learning_rate=learning_rate),
            'batch_norm_momentum': None
            },
    config=config
)

train_spec = tf.estimator.TrainSpec(
    input_fn=tf.compat.v1.estimator.inputs.numpy_input_fn(x={'f1': X_train},
                                                          y=y_train,
                                                          batch_size=batch_size,
                                                          num_epochs=None,
                                                          shuffle=False),
    max_steps=max_steps)

eval_spec = tf.estimator.EvalSpec(
    input_fn=tf.compat.v1.estimator.inputs.numpy_input_fn(x={'f1': X_test}, y=y_test,
                                                          num_epochs=None,
                                                          shuffle=False),
    throttle_secs=throttle_secs)

tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)
