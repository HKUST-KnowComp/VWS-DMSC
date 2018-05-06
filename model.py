import tensorflow as tf


def dmsc_fn(features, labels, mode, params):
    net = tf.feature_column.input_layer(features, params['feature_columns'])
