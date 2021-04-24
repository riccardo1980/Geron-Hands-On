import numpy as np
import tensorflow as tf


def neuron_layer(X, units, name, activation=None):
    with tf.name_scope(name):

        inputs = int(np.prod(X.get_shape().as_list()[1:]))
        stddev = 2 / np.sqrt(inputs + units)

        Z = tf.keras.layers.Dense(units,
                                  activation=activation,
                                  use_bias=True,
                                  kernel_initializer=tf.initializers.truncated_normal(
                                      stddev=stddev),
                                  bias_initializer='zeros')(X)
    return Z


def model_fn(features, labels, mode, params):

    # input
    net = tf.compat.v1.feature_column.input_layer(features,
                                                  params['feature_columns'])

    # hidden layers
    for idx, units in enumerate(params['hidden_units']):
        net = neuron_layer(net, units, name='dense_' +
                           str(idx+1), activation=params['activation'])

    # logits
    logits = neuron_layer(
        net, params['n_classes'], name='logits', activation=None)

    # predictions
    predicted_classes = tf.argmax(logits, 1)

    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = {
            'class_ids': predicted_classes[:, tf.newaxis],
            'probabilities': tf.keras.layers.Softmax(axis=1)(logits),
            'logits': logits
        }
        return tf.estimator.EstimatorSpec(mode, predictions=predictions)

    with tf.name_scope('loss'):
        xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels,
                                                                  logits=logits)
        loss = tf.reduce_mean(xentropy)

    with tf.name_scope('evaluation_metrics'):
        accuracy = tf.compat.v1.metrics.accuracy(labels, predicted_classes)

    metrics = {'accuracy': accuracy}
    tf.compat.v1.summary.scalar('accuracy', accuracy[1])

    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(mode,
                                          loss=loss,
                                          eval_metric_ops=metrics)

    assert mode == tf.estimator.ModeKeys.TRAIN
    optimizer = tf.compat.v1.train.AdagradOptimizer(
        learning_rate=params['learning_rate'])

    train_op = optimizer.minimize(
        loss, global_step=tf.compat.v1.train.get_global_step())
    return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)
