import numpy as np
import tensorflow.compat.v1 as tf

#pylint: disable=missing-docstring, C0301

def neuron_layer(X, units, name, mode, activation=None,
                 batch_norm_momentum=None):
    with tf.name_scope(name):

        inputs = int(np.prod(X.get_shape().as_list()[1:]))
        stddev = 2 / np.sqrt(inputs + units)

        if batch_norm_momentum is not None:
            # Dense + normalization + activation
            Z = tf.keras.layers.Dense(units,
                                      activation=None,
                                      use_bias=True,
                                      kernel_initializer=tf.initializers.truncated_normal(stddev=stddev),
                                      bias_initializer='zeros')(X)
            Z = tf.layers.batch_normalization(Z,
                                              training=mode == tf.estimator.ModeKeys.TRAIN,
                                              momentum=batch_norm_momentum)
            if activation is not None:
                Z = tf.keras.activations.get(activation)(Z)
        else:
            # Dense with activation
            Z = tf.keras.layers.Dense(units,
                                      activation=activation,
                                      use_bias=True,
                                      kernel_initializer=tf.initializers.truncated_normal(stddev=stddev),
                                      bias_initializer='zeros')(X)

        return Z

def model_fn(features, labels, mode, params):

    # input
    net = tf.feature_column.input_layer(features, params['feature_columns'])

    # input normalization
    if params['batch_norm_momentum'] is not None:
        net = tf.layers.batch_normalization(net,
                                            training=mode == tf.estimator.ModeKeys.TRAIN,
                                            momentum=params['batch_norm_momentum'],
                                            name='input_standardization')
    # hidden layers
    for idx, units in enumerate(params['hidden_units']):
        net = neuron_layer(net, units, name='dense_'+str(idx+1), mode=mode,
                           activation=params['activation'],
                           batch_norm_momentum=params['batch_norm_momentum'])

    # logits
    logits = neuron_layer(net, params['n_classes'], name='logits', mode=mode,
                          activation=None, batch_norm_momentum=params['batch_norm_momentum'])

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
    _input_fn = tf.estimator.inputs.numpy_input_fn(x={'features' : features},
                                                   y=labels,
                                                   batch_size=batch_size,
                                                   num_epochs=num_epochs,
                                                   shuffle=shuffle)
    return _input_fn

def serving_input_receiver_fn():
    inputs = {'features': tf.placeholder(shape=[None, 28, 28], dtype=tf.float32)}
    return tf.estimator.export.ServingInputReceiver(inputs, inputs)
