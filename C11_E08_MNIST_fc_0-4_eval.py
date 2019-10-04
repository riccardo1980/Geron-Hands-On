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

def get_graph(file_object, name=''):
    from google.protobuf import text_format
    graph_def = text_format.Parse(file_object.read(), tf.compat.v1.GraphDef())
    
    graph = tf.Graph()
    with graph.as_default():
        tf.import_graph_def(graph_def, name=name)
    
    return graph

def main(_):

    # load data
    train_data, test_data, y_train, y_test = get_data()

    # load estimator from checkpoint
    estimator = tf.contrib.estimator.SavedModelEstimator(FLAGS.model_dir)

    # with tf.compat.v1.Session() as sess:
    #     print("load graph")
        
    #     sess.graph.as_default()
    #     tf.import_graph_def(graph_def, name='')
    #     graph_nodes=[n for n in graph_def.node]
    #     names = []
    #     for t in graph_nodes:
    #         names.append(t.name)
    #     print(names)

    # detect whether there's a batch normalization just above input 'input_standardization'
        # if not: standardize test on train mean and var
    # if FLAGS.batch_norm_momentum is None:
        
    #     # explicit standardization: trained on train, reused on test dataset
    #     features_mean = np.mean(train_data, axis=0)
    #     features_std = np.std(train_data, axis=0)

    #     X_train = preproc_standardize(train_data, features_mean, features_std)
    #     X_test = preproc_standardize(test_data, features_mean, features_std)
    # else:
        
    #     # standardization is provided by a batch normalization layer
    #     X_train = train_data
    #     X_test = test_data
    
    # validate
    # predictions = estimator.predict(input_fn=tf.compat.v1.estimator.inputs.numpy_input_fn(x={'f1' : X_test},
    #                                                                                       y=y_test,
    #                                                                                       num_epochs=1,
    #                                                                                       shuffle=False))
    
    
    # y_pred = [pred['class_ids'] for pred in predictions]
    # print(classification_report(y_test, y_pred))

    # print(confusion_matrix(y_test, y_pred))

    # create feature columns
    pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.register("type", "bool", lambda v: v.lower() == "true")

    parser.add_argument(
        "--model_dir", type=str, required=True,
        help="Model dir."
    )

    FLAGS, unparsed = parser.parse_known_args()
    
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
    