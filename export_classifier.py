#!/usr/bin/env python
#-------------------------------------------------------------------------------
# Author: Lukasz Janyst <lukasz@jany.st>
# Date: 27.09.2017
#-------------------------------------------------------------------------------

import tensorflow as tf
from tensorflow.python.framework import graph_util

with tf.Session() as sess:
    tf.saved_model.loader.load(sess, ['tl_classifier'], 'models/classifier/')

    graph = tf.get_default_graph()
    input_graph_def = graph.as_graph_def()
    output_graph_def = graph_util.convert_variables_to_constants(
        sess, input_graph_def, ['predictions/prediction_class'])

    with open('cl-graph.pb', "wb") as f:
        f.write(output_graph_def.SerializeToString())
