# Copyright 2015 Paul Balanca. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Converts Pascal VOC data to TFRecords file format with Example protos.

The raw Pascal VOC data set is expected to reside in JPEG files located in the
directory 'JPEGImages'. Similarly, bounding box annotations are supposed to be
stored in the 'Annotation directory'

This TensorFlow script converts the training and evaluation data into
a sharded data set consisting of 1024 and 128 TFRecord files, respectively.

Each validation TFRecord file contains ~500 records. Each training TFREcord
file contains ~1000 records. Each record within the TFRecord file is a
serialized Example proto. The Example proto contains the following fields:

    image/encoded: string containing JPEG encoded image in RGB colorspace
    image/height: integer, image height in pixels
    image/width: integer, image width in pixels
    image/channels: integer, specifying the number of channels, always 3
    image/format: string, specifying the format, always'JPEG'


    image/object/bbox/xmin: list of float specifying the 0+ human annotated
        bounding boxes
    image/object/bbox/xmax: list of float specifying the 0+ human annotated
        bounding boxes
    image/object/bbox/ymin: list of float specifying the 0+ human annotated
        bounding boxes
    image/object/bbox/ymax: list of float specifying the 0+ human annotated
        bounding boxes
    image/object/bbox/label: list of integer specifying the classification index.
    image/object/bbox/label_text: list of string descriptions.

Note that the length of xmin is identical to the length of xmax, ymin and ymax
for each example.
"""
import os
import sys
import random

import numpy as np
import tensorflow as tf
import math

from datasets.dataset_utils import int64_feature, float_feature, bytes_feature
from datasets.pascalvoc_common import EVENT_LABELS

# Original dataset organisation.
DIRECTORY_ANNOTATIONS = 'Annotations/'
DIRECTORY_IMAGES = 'JPEGImages/'

# TFRecords convertion parameters.
RANDOM_SEED = 4242
SAMPLES_PER_FILES = 1000


def read_event_records(path_to_records, dataset_type):

    images = []
    data = []
    labels = []
    bbox_map = {}
    label_map = {}
    with open(os.path.join(path_to_records, "event_records.txt"), "r") as f:

        for l in f.readlines():
            l = l.replace("\n", "")
            tuples = l.split("\t")

            start_time = tuples[2]
            start_year = start_time[:4]

            if dataset_type == "event_train":
                if start_year == "2015":
                    continue
            elif dataset_type == "event_test":
                if start_year != "2015":
                    continue

            label = 0
            if tuples[1] == "AR":
                label = "ar"
            elif tuples[1] == "CH":
                label = "ch"
            else:
                continue

            bbox = tuples[4]
            bbox = [int(math.floor(float(i))) for i in bbox.split("-")]
            bbox.reverse()
            image_name = os.path.join(path_to_records, tuples[5] + "_171.jpg")
            if not image_name in bbox_map.keys():
                bbox_map[image_name] = [bbox]
                label_map[image_name] = [label]
            else:
                bbox_map[image_name].append(bbox)
                label_map[image_name].append(label)
    for image in bbox_map.keys():
        images.append(image)
        data.append(bbox_map[image])
        labels.append(label_map[image])
    return images, data, labels


def _process_image(filename, bboxes, labels):
    """Process a image and annotation file.

    Args:
      filename: string, path to an image file e.g., '/path/to/example.JPG'.
      coder: instance of ImageCoder to provide TensorFlow image coding utils.
    Returns:
      image_buffer: string, JPEG encoding of RGB image.
      height: integer, image height in pixels.
      width: integer, image width in pixels.
    """
    # Read the image file.
    image_data = tf.gfile.FastGFile(filename, 'r').read()

    if len(bboxes) != len(labels):
        raise ValueError("length of bboxes and labels are not same")

    # Read the XML annotation file.
    shape = [512, 512, 1]
    # Find annotations.
    labels = []
    labels_text = []
    difficult = []
    truncated = []
    new_bboxes = []
    for bbox in bboxes:
        new_bbox = [b / 4096 for b in bbox]
        new_bboxes.append(new_bbox)

    for label in labels:
        labels.append(int(EVENT_LABELS[label][0]))
        labels_text.append(label.encode('ascii'))
        difficult.append(0)
        truncated.append(0)
    return image_data, shape, new_bboxes, labels, labels_text, difficult, truncated


def _convert_to_example(image_data, labels, labels_text, bboxes, shape,
                        difficult, truncated):
    """Build an Example proto for an image example.

    Args:
      image_data: string, JPEG encoding of RGB image;
      labels: list of integers, identifier for the ground truth;
      labels_text: list of strings, human-readable labels;
      bboxes: list of bounding boxes; each box is a list of integers;
          specifying [xmin, ymin, xmax, ymax]. All boxes are assumed to belong
          to the same label as the image label.
      shape: 3 integers, image shapes in pixels.
    Returns:
      Example proto
    """
    xmin = []
    ymin = []
    xmax = []
    ymax = []
    for b in bboxes:
        assert len(b) == 4
        # pylint: disable=expression-not-assigned
        [l.append(point) for l, point in zip([ymin, xmin, ymax, xmax], b)]
        # pylint: enable=expression-not-assigned

    image_format = b'JPEG'
    example = tf.train.Example(features=tf.train.Features(feature={
            'image/height': int64_feature(shape[0]),
            'image/width': int64_feature(shape[1]),
            'image/channels': int64_feature(shape[2]),
            'image/shape': int64_feature(shape),
            'image/object/bbox/xmin': float_feature(xmin),
            'image/object/bbox/xmax': float_feature(xmax),
            'image/object/bbox/ymin': float_feature(ymin),
            'image/object/bbox/ymax': float_feature(ymax),
            'image/object/bbox/label': int64_feature(labels),
            'image/object/bbox/label_text': bytes_feature(labels_text),
            'image/object/bbox/difficult': int64_feature(difficult),
            'image/object/bbox/truncated': int64_feature(truncated),
            'image/format': bytes_feature(image_format),
            'image/encoded': bytes_feature(image_data)}))
    return example


def _add_to_tfrecord(name, bboxes, labels, tfrecord_writer):
    """Loads data from image and annotations files and add them to a TFRecord.

    Args:
      dataset_dir: Dataset directory;
      name: Image name to add to the TFRecord;
      tfrecord_writer: The TFRecord writer to use for writing.
    """
    image_data, shape, bboxes, labels, labels_text, difficult, truncated = \
        _process_image(name, bboxes, labels)
    example = _convert_to_example(image_data, labels, labels_text,
                                  bboxes, shape, difficult, truncated)
    tfrecord_writer.write(example.SerializeToString())


def _get_output_filename(output_dir, name, idx):
    return os.path.join(output_dir, '%s_%03d.tfrecord' % (name, idx))


def run(dataset_dir, output_dir, name='event_train', shuffling=False):
    """Runs the conversion operation.

    Args:
      dataset_dir: The dataset directory where the dataset is stored.
      output_dir: Output directory.
    """
    if not tf.gfile.Exists(dataset_dir):
        tf.gfile.MakeDirs(dataset_dir)

    images, data, labels = read_event_records(dataset_dir, dataset_type=name)
    # Process dataset files.
    i = 0
    fidx = 0
    while i < len(images):
        # Open new TFRecord file.
        tf_filename = _get_output_filename(output_dir, name, fidx)
        with tf.python_io.TFRecordWriter(tf_filename) as tfrecord_writer:
            j = 0
            while i < len(images) and j < SAMPLES_PER_FILES:
                sys.stdout.write('\r>> Converting image %d/%d' % (i+1, len(images)))
                sys.stdout.flush()

                image = images[i]
                _add_to_tfrecord(image, data[i], labels[i], tfrecord_writer)
                i += 1
                j += 1
            fidx += 1

    # Finally, write the labels file:
    # labels_to_class_names = dict(zip(range(len(_CLASS_NAMES)), _CLASS_NAMES))
    # dataset_utils.write_label_file(labels_to_class_names, dataset_dir)
    print('\nFinished converting the Pascal VOC dataset!')