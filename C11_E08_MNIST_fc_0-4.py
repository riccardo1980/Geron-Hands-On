import os
import sys
import warnings
import argparse
from datetime import datetime
import numpy as np
import json
from sklearn.metrics import classification_report, confusion_matrix

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=FutureWarning)
    import tensorflow as tf

    from C11_resources import FullyconnectedClassifier

def dataset_prune(features, labels, idx):
    return features[idx], labels[idx]

def get_data():
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

    return train_data, test_data, train_target, test_target

def preproc_standardize(dataset, mean, std):
    return (dataset - mean) / (std + 1e-15)


def main(_):

    train_data, test_data, y_train, y_test = get_data()

    if FLAGS.batch_norm_momentum is None:
        
        # explicit standardization: trained on train, reused on test dataset
        features_mean = np.mean(train_data, axis=0)
        features_std = np.std(train_data, axis=0)

        X_train = preproc_standardize(train_data, features_mean, features_std)
        X_test = preproc_standardize(test_data, features_mean, features_std)
    else:
        
        # standardization is provided by a batch normalization layer
        X_train = train_data
        X_test = test_data

    
    n_classes = len(set(y_train))

    print('n_classes: {}'.format(n_classes))
    print('epochs: {}'.format(FLAGS.max_steps * FLAGS.train_batch_size / y_train.shape[0]))

    # this seed is used only for initialization
    # batch is still random with no chance to set the seed
    # see: https://stackoverflow.com/questions/47009560/tf-estimator-shuffle-random-seed
    config = tf.estimator.RunConfig(tf_random_seed=42,
                                    model_dir=FLAGS.model_dir,
                                    save_checkpoints_steps=FLAGS.save_checkpoints_steps,
                                    log_step_count_steps=FLAGS.log_step_count_steps)

    # create feature columns
    feature_columns = [tf.feature_column.numeric_column(key='f1', shape=X_train[0].shape)]

    params = {'feature_columns': feature_columns,
              'hidden_units': FLAGS.hidden_units,
              'activation': 'elu',
              'n_classes': n_classes,
              'optimizer': tf.compat.v1.train.AdagradOptimizer(learning_rate=FLAGS.learning_rate),
              'batch_norm_momentum': FLAGS.batch_norm_momentum
             }

    estimator = tf.estimator.Estimator(
        model_fn=FullyconnectedClassifier.model_fn,
        params=params,
        config=config
        )

    train_spec = tf.estimator.TrainSpec(
        input_fn=tf.compat.v1.estimator.inputs.numpy_input_fn(x={'f1' : X_train},
                                                              y=y_train,
                                                              batch_size=FLAGS.train_batch_size,
                                                              num_epochs=None,
                                                              shuffle=False),
        max_steps=FLAGS.max_steps)

    eval_spec = tf.estimator.EvalSpec(
        input_fn=tf.compat.v1.estimator.inputs.numpy_input_fn(x={'f1' : X_test},
                                                              y=y_test,
                                                              num_epochs=1,
                                                              shuffle=False),
        throttle_secs=FLAGS.throttle_secs)

    tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)

    # export
    serving_input_receiver_fn = (
        tf.estimator.export.build_parsing_serving_input_receiver_fn(            
            tf.feature_column.make_parse_example_spec(feature_columns)))

    export_dir = estimator.export_saved_model(os.path.join(FLAGS.model_dir,'saved_model'),
                                              serving_input_receiver_fn)
    
    print('Model exported in: {}'.format(export_dir))

    # validate
    predictions = estimator.predict(input_fn=tf.compat.v1.estimator.inputs.numpy_input_fn(x={'f1' : X_test},
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
        "--batch_norm_momentum", type=float, default=None,
        help="Batch norm momentum."
    )

    parser.add_argument(
        "--model_dir", type=str,
        default=os.path.join('./tmp',datetime.utcnow().strftime('%Y%m%d-%H%M%S')),
        help="Model dir."
    )
    
    parser.add_argument(
        "--hidden_units", type=str,
        default="[100, 100, 100, 100, 100]",
        help="Array of hidden units."
    )

    FLAGS, unparsed = parser.parse_known_args()
    
    # transform strings to arrays
    FLAGS.hidden_units=json.loads(FLAGS.hidden_units)
    
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
    