import os
import sys
import warnings
import argparse
from datetime import datetime
import json
import numpy as np
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

# def get_graph(file_object, name=''):
#     from google.protobuf import text_format
#     graph_def = text_format.Parse(file_object.read(), tf.compat.v1.GraphDef())

#     graph = tf.Graph()
#     with graph.as_default():
#         tf.import_graph_def(graph_def, name=name)

#     return graph

def main(_):

    # load data
    _, X_test, _, y_test = get_data()

    # load predictor from checkpoint
    predict_fn = tf.contrib.predictor.from_saved_model(FLAGS.saved_model_dir)

    # validate
    predictions = predict_fn({'features': X_test})

    y_pred = predictions['class_ids']
    print(classification_report(y_test, y_pred))

    print(confusion_matrix(y_test, y_pred))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.register("type", "bool", lambda v: v.lower() == "true")

    parser.add_argument(
        "--saved_model_dir", type=str, required=True,
        help="Saved model dir."
    )

    FLAGS, unparsed = parser.parse_known_args()

    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
