
import math
import os
import tensorflow as tf
from datasets import dataset_utils

slim = tf.contrib.slim


ITEMS_TO_DESCRIPTIONS = {
	'image': 'A color image of varying height and width.',
	'shape': 'Shape of the image',
	'object/bbox': 'A list of bounding boxes, one per each object.',
	'object/label': 'A list of labels, one per each object.',
}

# TRAIN_STATISTICS = {
# 	'none': (0, 0),
# 	'ar': (13035, 54796),
# 	'ch': (13035, 43081),
# }
SPLITS_TO_SIZES = {
	'train': 10337,
	'test': 2698,
}
# SPLITS_TO_STATISTICS = {
# 	'train': TRAIN_STATISTICS,
# }
NUM_CLASSES = 3

def get_split(split_name, dataset_dir, file_pattern, reader):
	"""Gets a dataset tuple with instructions for reading Pascal VOC dataset.

	Args:
	  split_name: A train/test split name.
	  dataset_dir: The base directory of the dataset sources.
	  file_pattern: The file pattern to use when matching the dataset sources.
		It is assumed that the pattern contains a '%s' string so that the split
		name can be inserted.
	  reader: The TensorFlow reader type.

	Returns:
	  A `Dataset` namedtuple.

	Raises:
		ValueError: if `split_name` is not a valid train/test split.
	"""
	if split_name not in SPLITS_TO_SIZES:
		raise ValueError('split name %s was not recognized.' % split_name)
	file_pattern="event_%s_000.tfrecord"
	file_pattern = os.path.join(dataset_dir, file_pattern % split_name)

	# Allowing None in the signature so that dataset_factory can use the default.
	if reader is None:
		reader = tf.TFRecordReader
	# Features in Pascal VOC TFRecords.
	keys_to_features = {
		'image/encoded': tf.FixedLenFeature((), tf.string, default_value=''),
		'image/format': tf.FixedLenFeature((), tf.string, default_value='jpeg'),
		'image/height': tf.FixedLenFeature([1], tf.int64),
		'image/width': tf.FixedLenFeature([1], tf.int64),
		'image/channels': tf.FixedLenFeature([1], tf.int64),
		'image/shape': tf.FixedLenFeature([3], tf.int64),
		'image/object/bbox/xmin': tf.VarLenFeature(dtype=tf.float32),
		'image/object/bbox/ymin': tf.VarLenFeature(dtype=tf.float32),
		'image/object/bbox/xmax': tf.VarLenFeature(dtype=tf.float32),
		'image/object/bbox/ymax': tf.VarLenFeature(dtype=tf.float32),
		'image/object/bbox/label': tf.VarLenFeature(dtype=tf.int64),
		'image/object/bbox/difficult': tf.VarLenFeature(dtype=tf.int64),
		'image/object/bbox/truncated': tf.VarLenFeature(dtype=tf.int64),
	}
	items_to_handlers = {
		'image': slim.tfexample_decoder.Image('image/encoded', 'image/format', channels=1),
		'shape': slim.tfexample_decoder.Tensor('image/shape'),
		'object/bbox': slim.tfexample_decoder.BoundingBox(
			['ymin', 'xmin', 'ymax', 'xmax'], 'image/object/bbox/'),
		'object/label': slim.tfexample_decoder.Tensor('image/object/bbox/label'),
		'object/difficult': slim.tfexample_decoder.Tensor('image/object/bbox/difficult'),
		'object/truncated': slim.tfexample_decoder.Tensor('image/object/bbox/truncated'),
	}
	decoder = slim.tfexample_decoder.TFExampleDecoder(
		keys_to_features, items_to_handlers)

	labels_to_names = None
	if dataset_utils.has_labels(dataset_dir):
		labels_to_names = dataset_utils.read_label_file(dataset_dir)
	# else:
	#     labels_to_names = create_readable_names_for_imagenet_labels()
	#     dataset_utils.write_label_file(labels_to_names, dataset_dir)

	return slim.dataset.Dataset(
		data_sources=file_pattern,
		reader=reader,
		decoder=decoder,
		num_samples=SPLITS_TO_SIZES[split_name],
		items_to_descriptions=ITEMS_TO_DESCRIPTIONS,
		num_classes=NUM_CLASSES,
		labels_to_names=labels_to_names)