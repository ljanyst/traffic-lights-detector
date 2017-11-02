#!/usr/bin/env python
#-------------------------------------------------------------------------------
# Author: Lukasz Janyst <lukasz@jany.st>
# Date: 27.09.2017
#-------------------------------------------------------------------------------

import PIL.Image
import hashlib
import logging
import random
import yaml
import cv2
import os
import io

import tensorflow as tf

from object_detection.utils import label_map_util
from object_detection.utils import dataset_util
from tqdm import tqdm
from glob import glob

#-------------------------------------------------------------------------------
# Set the app flags up
#-------------------------------------------------------------------------------
flags = tf.app.flags
flags.DEFINE_string('data_dir', 'data', 'Root directory to raw pet dataset.')
flags.DEFINE_string('output_dir', 'data', 'Path to directory to output TFRecords.')
flags.DEFINE_string('label_map_path', 'lights_label_map.pbtxt',
                    'Path to label map proto')
FLAGS = flags.FLAGS

#-------------------------------------------------------------------------------
def dict_to_tf_example(data):
    #---------------------------------------------------------------------------
    # Read the JPEG
    #---------------------------------------------------------------------------
    with open(data['filename'], 'rb') as f:
        encoded_jpg = f.read()
    encoded_jpg_io = io.BytesIO(encoded_jpg)
    image = PIL.Image.open(encoded_jpg_io)
    if image.format != 'JPEG':
        raise ValueError('Image format not JPEG')
    key = hashlib.sha256(encoded_jpg).hexdigest()

    #---------------------------------------------------------------------------
    # Process the image and box metadata
    #---------------------------------------------------------------------------
    width = int(data['width'])
    height = int(data['height'])

    xmin = []
    ymin = []
    xmax = []
    ymax = []
    classes = []
    classes_text = []

    for obj in data['boxes']:
        xmin.append(float(obj['xmin']) / width)
        ymin.append(float(obj['ymin']) / height)
        xmax.append(float(obj['xmax']) / width)
        ymax.append(float(obj['ymax']) / height)
        classes_text.append('traffic-lights'.encode('utf8'))
        classes.append(1)

    #---------------------------------------------------------------------------
    # Create the example object
    #---------------------------------------------------------------------------
    example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(height),
        'image/width': dataset_util.int64_feature(width),
        'image/filename': dataset_util.bytes_feature(
            data['filename'].encode('utf8')),
        'image/source_id': dataset_util.bytes_feature(
            data['filename'].encode('utf8')),
        'image/key/sha256': dataset_util.bytes_feature(key.encode('utf8')),
        'image/encoded': dataset_util.bytes_feature(encoded_jpg),
        'image/format': dataset_util.bytes_feature('jpeg'.encode('utf8')),
        'image/object/bbox/xmin': dataset_util.float_list_feature(xmin),
        'image/object/bbox/xmax': dataset_util.float_list_feature(xmax),
        'image/object/bbox/ymin': dataset_util.float_list_feature(ymin),
        'image/object/bbox/ymax': dataset_util.float_list_feature(ymax),
        'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
        'image/object/class/label': dataset_util.int64_list_feature(classes),
    }))
    return example

#-------------------------------------------------------------------------------
def create_tf_record(output_filename, examples):
    writer = tf.python_io.TFRecordWriter(output_filename)
    for idx, example in enumerate(examples):
        if idx % 100 == 0:
            logging.info('On image %d of %d', idx, len(examples))
        tf_example = dict_to_tf_example(example)
        writer.write(tf_example.SerializeToString())

    writer.close()

#-------------------------------------------------------------------------------
def build_sample_list_bosch(data_dir):
    #---------------------------------------------------------------------------
    # Read the metadata
    #---------------------------------------------------------------------------
    with open(data_dir+'/train.yaml', 'r') as f:
        try:
            records = yaml.load(f)
        except yaml.YAMLError as e:
            raise RuntimeError(e)

    #---------------------------------------------------------------------------
    # Loop over the records
    #---------------------------------------------------------------------------
    samples = []
    for record in tqdm(records, desc='bosch', unit='samples'):
        filename = data_dir+'/'+record['path'][:-3]+'jpg'
        if not os.path.exists(filename):
            continue

        img = cv2.imread(filename)
        width = img.shape[1]
        height = img.shape[0]

        #-----------------------------------------------------------------------
        # Decode all the boxes
        #-----------------------------------------------------------------------
        boxes = []
        for box_info in record['boxes']:
            if box_info['occluded'] or box_info['label'] == 'off':
                continue

            xmin = box_info['x_min']
            xmax = box_info['x_max']
            ymin = box_info['y_min']
            ymax = box_info['y_max']

            if ymax - ymin < 0.015*height or xmax - xmin < 0.015*width:
                continue

            box = {
                'xmin': xmin,
                'xmax': xmax,
                'ymin': ymin,
                'ymax': ymax
            }
            boxes.append(box)

        #-----------------------------------------------------------------------
        # Encode the sample
        #-----------------------------------------------------------------------
        if not boxes:
            continue

        sample = {
            'filename': filename,
            'boxes': boxes,
            'width': width,
            'height': height
        }
        samples.append(sample)

    return samples

#-------------------------------------------------------------------------------
def build_sample_list_udacity(data_dir):
    #---------------------------------------------------------------------------
    # Read the meatdata
    #---------------------------------------------------------------------------
    files = []
    root = data_dir+'/udacity-boxes/'
    for name in ['sim_images', 'site_images']:
        files += glob(root+name+'/*.txt')

    #---------------------------------------------------------------------------
    # Loop over the records
    #---------------------------------------------------------------------------
    samples = []
    for filename in tqdm(files, desc='udacity', unit='samples'):
        if not os.path.exists(filename):
            continue

        with open(filename, 'r') as f:
            data = f.readlines()[1:]

        data = list(map(lambda x: x.split(), data))

        img_filename = '.'.join(os.path.basename(filename).split('.')[:-1])
        img_filename = os.path.dirname(filename)+'/'+img_filename+'.jpg'
        if not os.path.exists(img_filename):
            continue

        img = cv2.imread(img_filename)
        width = img.shape[1]
        height = img.shape[0]

        #-----------------------------------------------------------------------
        # Decode the boxes
        #-----------------------------------------------------------------------
        boxes = []
        for box_info in data:
            xmin = int(box_info[0])
            xmax = int(box_info[2])
            ymin = int(box_info[1])
            ymax = int(box_info[3])
            box = {
                'xmin': xmin,
                'xmax': xmax,
                'ymin': ymin,
                'ymax': ymax
            }
            boxes.append(box)

        #-----------------------------------------------------------------------
        # Encode the sample
        #-----------------------------------------------------------------------
        if not boxes:
            continue

        sample = {
            'filename': img_filename,
            'boxes': boxes,
            'width': width,
            'height': height
        }
        samples.append(sample)

    return samples

#-------------------------------------------------------------------------------
def main(_):
    #---------------------------------------------------------------------------
    # Read the data
    #---------------------------------------------------------------------------
    label_map_dict = label_map_util.get_label_map_dict(FLAGS.label_map_path)
    bosch = build_sample_list_bosch(FLAGS.data_dir)
    udacity = build_sample_list_udacity(FLAGS.data_dir)
    print(len(bosch))
    samples = bosch + udacity

    #---------------------------------------------------------------------------
    # Split into training and validation
    #---------------------------------------------------------------------------
    random.shuffle(samples)
    num_examples = len(samples)
    num_train = int(0.975 * num_examples)
    train_examples = samples[:num_train]
    val_examples = samples[num_train:]
    logging.info('%d training and %d validation examples.',
                 len(train_examples), len(val_examples))

    #---------------------------------------------------------------------------
    # Create the record files
    #---------------------------------------------------------------------------
    train_output_path = os.path.join(FLAGS.output_dir, 'lights_train.record')
    val_output_path = os.path.join(FLAGS.output_dir, 'lights_val.record')
    create_tf_record(train_output_path, train_examples)
    create_tf_record(val_output_path, val_examples)

#-------------------------------------------------------------------------------
if __name__ == '__main__':
  tf.app.run()
