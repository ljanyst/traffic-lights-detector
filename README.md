
traffic-lights-detector
=======================

This repository contains the training code for the traffic lights detector
used by the [Kung-Fu-Panda][1] team for the [Udacity SDCND][2] Capstone Project.
The detector is based on the [TensorFlow object detection API][3] and uses
[the model][4] produced by Alexey Simonov (our ex-team member) for
classification.

Installation
------------

You will need the TensorFlow models repository and a bunch of other things to
run this code. Follow [these installation instructions][5] for a detailed
explanation. Assuming you already have all the python dependencies installed,
the set-up process may be summarised as follows:

    sudo apt-get install protobuf-compiler
    git clone git@github.com:tensorflow/models.git
    cd models/research
    protoc object_detection/protos/*.proto --python_out=.
    export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim

Training Data
-------------

The `create_sign_tf_record.py` program parses the input datasets and
produces input files for the object detection trainer. As far as the data
is concerned, I used the [Bosch Small Traffic Lights][6] and a bunch of
hand-annotated images extracted from the [ROSBags][7] provided by Udacity
and recorded from [the simulator][8]. I used [this tool][9] to create
annotations. The script rejects the images from the Bosch datasets containing
only very small objects.

    cd data
    unzip dataset_train_rgb.zip
    tar zxf udacity-boxes.tar.gz

You'll need to convert the Bosch images to jpegs. I used *ImageMagick* and
shell:

    for i in *png; do convert $i `basename $i .png`.jpg; rm -f $i; done

Training
--------

First, you will need to download the base model:

    mkdir model
    cd model
    wget http://download.tensorflow.org/models/object_detection/faster_rcnn_resnet101_coco_11_06_2017.tar.gz
    tar zxf faster_rcnn_resnet101_coco_11_06_2017.tar.gz
    cd ..

Then, you can train the model:

    python $MODELS/research/object_detection/train.py  \
        --logtostderr \
        --pipeline_config_path=faster_rcnn_resnet101_lights.config
        --train_dir=train

I find that running it for about 30000 iterations produces pretty good results.

Exporting the graph
-------------------

The most convenient way to use multiple models in a single application is to
export them as static inference graphs. For the detection model, you can just
use the utility provided by the TensorFlow Object Detection API:

    python  $MODELS/research/object_detection/export_inference_graph.py \
        --input_type image_tensor \
        --pipeline_config_path faster_rcnn_resnet101_lights.config \
        --trained_checkpoint_prefix train/model.ckpt-30452 \
        --output_directory output_inference_graph.pb

I wrote a small utility program (`export_classifier.py`) to export the
classification model created by Alexey.

Results
-------

I wrote a small testing script that takes a bunch of images as command-line
parameters and runs them, first, through the detector, and then, through
the classifier. `ffmpeg` may then be used to create a video from multiple
frames:

    ./detect.py data/site/*.jpg
    ffmpeg -framerate 24 -i output/left%04d.jpg output.mp4

You can watch sample videos by clicking on the images below:

[![Traffic Light Detection - Site](https://img.youtube.com/vi/rB9oImCdhMQ/0.jpg)](https://www.youtube.com/watch?v=rB9oImCdhMQ "Traffic Light Detection - Site")
[![Traffic Light Detection - Simulator](https://img.youtube.com/vi/Rt6u1fywLzI/0.jpg)](https://www.youtube.com/watch?v=Rt6u1fywLzI "Traffic Light Detection - Simulator")

[1]: https://github.com/kung-fu-panda-automotive/carla-driver
[2]: https://www.udacity.com/course/self-driving-car-engineer-nanodegree--nd013
[3]: https://github.com/tensorflow/models/tree/master/research/object_detection
[4]: https://github.com/asimonov/Bosch-TL-Dataset
[5]: https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/installation.md
[6]: https://hci.iwr.uni-heidelberg.de/node/6132
[7]: https://drive.google.com/file/d/0B2_h37bMVw3iYkdJTlRSUlJIamM/view
[8]: https://github.com/udacity/CarND-Capstone/releases/tag/v1.2
[9]: https://github.com/ljanyst/bbox-label-tool
