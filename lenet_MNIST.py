import os
import sys
import warnings
import argparse
from datetime import datetime
import json
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix


from tensorflow.python.util import deprecation as tf_deprecation
tf_deprecation._PRINT_DEPRECATION_WARNINGS = False
import tensorflow.compat.v1 as tf
from tensorflow.compat.v1.train import AdagradOptimizer
from tensorflow.compat.v1.feature_column import input_layer 
from tensorflow.compat.v1.keras.layers import Conv2D, MaxPool2D, Dense, Flatten
from tensorflow.compat.v1 import variance_scaling_initializer

#pylint: disable=missing-docstring, C0301

def model_fn_LENET_5(features,
                     activation = 'relu',
                     kernel_initializer = tf.keras.initializers.TruncatedNormal(mean=0, stddev=0.1),
                     bias_initializer = 'zeros'):

    # conv1: output is [None, 28, 28, 6]    
    conv1 = Conv2D(filters=6, kernel_size=(5,5), strides=(1,1), 
                   padding='valid', activation=activation, use_bias=True,
                   kernel_initializer=kernel_initializer, bias_initializer=bias_initializer)(features)

    # pool1: output is [None, 14, 14, 6]
    pool1 = MaxPool2D(pool_size=(2,2))(conv1)

    # conv2: output is [None, 10, 10, 16]
    conv2 = Conv2D(filters=16, kernel_size=(5,5), strides=(1,1),
                   padding='valid', activation=activation, use_bias=True,
                   kernel_initializer=kernel_initializer, bias_initializer=bias_initializer)(pool1)

    # pool2: output is [None, 5, 5, 16] -> flattened on input of FC to [None, 400]
    pool2 = MaxPool2D(pool_size=(2,2))(conv2)
    flatten = Flatten()(pool2)

    # fc3: output is [None, 120]
    fc3 = Dense(units=120, activation=activation, use_bias=True,
                kernel_initializer=kernel_initializer,
                bias_initializer=bias_initializer)(flatten)

    # fc4: output is [None, 84]
    fc4 = Dense(units=84, activation=activation, use_bias=True,
                kernel_initializer=kernel_initializer,
                bias_initializer=bias_initializer)(fc3)

    return fc4


def model_fn(features, labels, mode, params):

    kernel_initializer = variance_scaling_initializer(distribution='truncated_normal')
    #kernel_initializer = tf.keras.initializers.TruncatedNormal(mean=0, stddev=0.1)
    bias_initializer = 'zeros'

    # input: [None, 32, 32, 1]
    feat = tf.reshape(input_layer(features, params['feature_columns']), [-1, 32, 32, 1])

    net = model_fn_LENET_5(feat, activation='relu',
                           kernel_initializer=kernel_initializer,
                           bias_initializer=bias_initializer)

    # logits: output is [None, CLASSES]
    logits = Dense(units=params['n_classes'], activation=None, use_bias=True,
                    kernel_initializer=kernel_initializer,
                    bias_initializer=bias_initializer)(net)

    # predictions
    predicted_classes = tf.argmax(logits, 1)

    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = {
            'class_ids': predicted_classes[:, tf.newaxis],
            'probabilities': tf.keras.layers.Softmax(axis=1)(logits),
            'logits': logits
        }
        return tf.estimator.EstimatorSpec(mode, predictions=predictions)

    xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels,
                                                              logits=logits)
    loss = tf.reduce_mean(xentropy)

    accuracy = tf.metrics.accuracy(labels, predicted_classes, name='acc_op')

    with tf.name_scope('metrics'):
        tf.summary.scalar('accuracy', accuracy[1])

    metrics = {
        'metrics/accuracy': accuracy,
    }

    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(mode,
                                          loss=loss,
                                          eval_metric_ops=metrics)

    assert mode == tf.estimator.ModeKeys.TRAIN
    optimizer = params['optimizer']

    # get operations related to batch normalization
    # see: https://stackoverflow.com/questions/45299522/batch-normalization-in-a-custom-estimator-in-tensorflow
    # see: https://github.com/tensorflow/tensorflow/issues/16455
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())

    return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)

def make_input_fn(features, labels=None, batch_size=128, num_epochs=1, shuffle=False):
    print('\nSHAPE: {}\n'.format(features.shape))
    _input_fn = tf.estimator.inputs.numpy_input_fn(x={'features' : features},
                                                   y=labels,
                                                   batch_size=batch_size,
                                                   num_epochs=num_epochs,
                                                   shuffle=shuffle)
    return _input_fn

def serving_input_receiver_fn():
    inputs = {'features': tf.placeholder(shape=[None, 32, 32, 1], dtype=tf.float32)}
    return tf.estimator.export.ServingInputReceiver(inputs, inputs)

def get_data():
    # read complete dataset
    (train_data, train_target), (test_data, test_target) = tf.keras.datasets.mnist.load_data()

    # cast features to float32, targets to int32
    train_data = train_data.astype(np.float32)
    test_data = test_data.astype(np.float32)
    train_target = train_target.astype(np.int32)
    test_target = test_target.astype(np.int32)
    
    # Pad images with 0s in WH of BWHC
    train_data = np.pad(np.expand_dims(train_data, 3), ((0,0),(2,2),(2,2),(0,0)), 'constant')
    test_data = np.pad(np.expand_dims(test_data, 3), ((0,0),(2,2),(2,2),(0,0)), 'constant')
    
    return train_data, test_data, train_target, test_target


def main(_):

    train_data, test_data, y_train, y_test = get_data()
    
    n_classes = len(set(y_train))

    print('\ntrain set size: {}'.format(train_data.shape ))
    print('example size: {}'.format(train_data.shape[1:] ))
    print('n_classes: {}'.format(n_classes))
    print('epochs: {}\n'.format(FLAGS.max_steps * FLAGS.train_batch_size / y_train.shape[0]))

    # this seed is used only for initialization
    # batch is still random with no chance to set the seed
    # see: https://stackoverflow.com/questions/47009560/tf-estimator-shuffle-random-seed
    config = tf.estimator.RunConfig(tf_random_seed=42,
                                    model_dir=FLAGS.model_dir,
                                    save_checkpoints_steps=FLAGS.save_checkpoints_steps,
                                    log_step_count_steps=FLAGS.log_step_count_steps)

    # create feature columns
    feature_columns = [tf.feature_column.numeric_column(key='features', shape=train_data.shape[1:])]

    params = {'feature_columns': feature_columns,
              'n_classes': n_classes,
              'optimizer': AdagradOptimizer(learning_rate=FLAGS.learning_rate)
             }

    classifier = tf.estimator.Estimator(
        model_fn=model_fn,
        params=params,
        config=config
        )

    train_spec = tf.estimator.TrainSpec(
        input_fn=make_input_fn(train_data,
                               labels=y_train,
                               batch_size=FLAGS.train_batch_size,
                               num_epochs=None,
                               shuffle=False),
        max_steps=FLAGS.max_steps)

    eval_spec = tf.estimator.EvalSpec(
        input_fn=make_input_fn(test_data, labels=y_test),
        throttle_secs=FLAGS.throttle_secs)

    # training
    tf.estimator.train_and_evaluate(classifier, train_spec, eval_spec)

    #export
    export_dir = classifier.export_saved_model(os.path.join(FLAGS.model_dir, 'saved_model'),
                                               serving_input_receiver_fn=serving_input_receiver_fn)

    print('Model exported in: {}'.format(export_dir))

    # validation
    predictions = classifier.predict(input_fn=tf.estimator.inputs.numpy_input_fn(x={'features' : test_data},
                                                                                 y=y_test,
                                                                                 num_epochs=1,
                                                                                 shuffle=False))

    y_pred = [pred['class_ids'] for pred in predictions]
    print(classification_report(y_test, y_pred))

    print(confusion_matrix(y_test, y_pred))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.register("type", "bool", lambda v: v.lower() == "true")

    parser.add_argument(
        "--max_steps", type=int, default=30000,
        help="Number of steps to run trainer."
    )

    parser.add_argument(
        "--train_batch_size", type=int, default=200,
        help="Batch size used during training."
    )

    parser.add_argument(
        "--learning_rate", type=float, default=1e-4,
        help="Initial learning rate."
    )

    parser.add_argument(
        "--save_checkpoints_steps", type=int, default=2000,
        help="Save checkpoints every this many steps."
    )

    parser.add_argument(
        "--log_step_count_steps", type=int, default=100,
        help="Log and summary frequency, in global steps."
    )

    parser.add_argument(
        "--throttle_secs", type=int, default=10,
        help="Evaluation throttle in seconds."
    )

    parser.add_argument(
        "--model_dir", type=str,
        default=os.path.join('./tmp', datetime.utcnow().strftime('%Y%m%d-%H%M%S')),
        help="Model dir."
    )

    FLAGS, UNPARSED = parser.parse_known_args()

    tf.app.run(main=main, argv=[sys.argv[0]] + UNPARSED)