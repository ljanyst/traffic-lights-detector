#!/usr/bin/env python
#-------------------------------------------------------------------------------
# Author: Lukasz Janyst <lukasz@jany.st>
# Date: 27.09.2017
#-------------------------------------------------------------------------------

import os
import sys
import cv2

import numpy as np
import tensorflow as tf

from collections import namedtuple
from tqdm import tqdm

Box = namedtuple('Box', ['score', 'xmin', 'xmax', 'ymin', 'ymax'])
Size = namedtuple('Size', ['w', 'h'])

#-------------------------------------------------------------------------------
# Decode box
#-------------------------------------------------------------------------------
def decode_boxes(img_size, boxes, scores, threshold):
    scores = scores[scores>threshold]
    dec_boxes = []
    for i, box in enumerate(boxes[:len(scores)]):
        dec_box = Box(scores[i],
                      int(img_size.w*box[1]), int(img_size.w*box[3]),
                      int(img_size.h*box[0]), int(img_size.h*box[2]))
        dec_boxes.append(dec_box)
    return dec_boxes

#-------------------------------------------------------------------------------
# Draw box
#-------------------------------------------------------------------------------
def draw_box(img, box, color, text):
    cv2.rectangle(img, (box.xmin, box.ymin), (box.xmax, box.ymax), color, 2)
    cv2.rectangle(img, (box.xmin-1, box.ymin), (box.xmax+1, box.ymin-20), color, cv2.FILLED)
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img, text, (box.xmin+5, box.ymin-5), font, 0.5,
                (255, 255, 255), 1, cv2.LINE_AA)

#-------------------------------------------------------------------------------
# Light metadata
#-------------------------------------------------------------------------------
light_labels = ['off', 'green', 'yellow', 'red']
light_colors = [
    ( 70,  70,  70),
    ( 52, 151,  52),
    (  0, 220, 220),
    ( 60,  20, 220)]

def main():
    #---------------------------------------------------------------------------
    # Read the detector metagraph
    #---------------------------------------------------------------------------
    detector_graph_def = tf.GraphDef()
    with open('models/traffic-lights-faster-r-cnn-new-1.pb', 'rb') as f:
        serialized = f.read()
        detector_graph_def.ParseFromString(serialized)

    #---------------------------------------------------------------------------
    # Read the classifier metagraph
    #---------------------------------------------------------------------------
    classifier_graph_def = tf.GraphDef()
    with open('models/traffic-lights-classifier.pb', 'rb') as f:
        serialized = f.read()
        classifier_graph_def.ParseFromString(serialized)

    with tf.Session() as sess:
        #-----------------------------------------------------------------------
        # Set the detector up
        #-----------------------------------------------------------------------
        tf.import_graph_def(detector_graph_def, name='detector')
        detection_input = sess.graph.get_tensor_by_name('detector/image_tensor:0')
        detection_boxes = sess.graph.get_tensor_by_name('detector/detection_boxes:0')
        detection_scores = sess.graph.get_tensor_by_name('detector/detection_scores:0')

        #-----------------------------------------------------------------------
        # Set the classifier up
        #-----------------------------------------------------------------------
        tf.import_graph_def(classifier_graph_def, name='classifier')
        classifier_input = sess.graph.get_tensor_by_name('classifier/data/images:0')
        classifier_prediction = sess.graph.get_tensor_by_name('classifier/predictions/prediction_class:0')
        classifier_keep_prob = sess.graph.get_tensor_by_name('classifier/dropout_keep_probability:0')

        try:
            os.makedirs('output')
        except IOError:
            pass

        for in_file in tqdm(sys.argv[1:]):
            out_file = 'output/'+os.path.basename(in_file)

            #-------------------------------------------------------------------
            # Detect boxes
            #-------------------------------------------------------------------
            img = cv2.cvtColor(cv2.imread(in_file), cv2.COLOR_BGR2RGB)
            img_expanded = np.expand_dims(img, axis=0)

            boxes, scores = sess.run([detection_boxes, detection_scores],
                                     feed_dict={detection_input: img_expanded})

            img_size = Size(img.shape[1], img.shape[0])
            detected_boxes = decode_boxes(img_size, boxes[0], scores[0], 0.9)

            #-------------------------------------------------------------------
            # Classify the boxes
            #-------------------------------------------------------------------
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

            boxes = []
            for box in detected_boxes:
                img_light = img[box.ymin:box.ymax, box.xmin:box.xmax]
                img_light = cv2.resize(img_light, (32, 32))
                img_light_expanded = np.expand_dims(img_light, axis=0)
                light_class = sess.run(classifier_prediction,
                                       feed_dict={
                                           classifier_input: img_light_expanded,
                                           classifier_keep_prob: 1.})
                boxes.append((box, light_class[0]))

            #-------------------------------------------------------------------
            # Draw the boxes
            #-------------------------------------------------------------------
            for box, light_class in boxes:
                color = light_colors[light_class]
                label = light_labels[light_class]
                draw_box(img, box, color, label)

            cv2.imwrite(out_file, img)

if __name__ == '__main__':
    sys.exit(main())
