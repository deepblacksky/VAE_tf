# coding=utf-8
from __future__ import print_function
from __future__ import division

import tensorflow as tf
import os


class MnistReader(object):
    def __init__(self, filepath, batch_size, is_training=True):
        tf.gfile.Copy(filepath, os.path.join(os.getcwd(), 'temp.tfrecords'), overwrite=True)
        train_file_path = os.path.join(os.getcwd(), 'temp.tfrecords')
        self.filename_queue = tf.train.string_input_producer(tf.train.match_filenames_once(train_file_path))
        self.is_training = is_training
        if self.is_training:
            self.batch_size = batch_size
        else:
            self.batch_size = 5

    @staticmethod
    def read_image(file_queue):
        reader = tf.TFRecordReader()
        _, file_serialized_content = reader.read(file_queue)
        features = tf.parse_single_example(serialized=file_serialized_content,
                                           features={
                                               'image_raw': tf.FixedLenFeature([], tf.string),
                                               'label': tf.FixedLenFeature([], tf.int64)
                                           })

        image = tf.decode_raw(features['image_raw'], tf.uint8)
        image.set_shape([28*28])
        # normalization
        image = tf.cast(image, tf.float32) * (1. / 255)
        label = tf.cast(features['label'], tf.int32)
        return image, label

    def read_image_batch(self):
        img, label = self.read_image(self.filename_queue)
        image_batch = tf.train.batch([img], batch_size=self.batch_size,
                                     num_threads=4,
                                     capacity=1000 + self.batch_size * 3)
        # because VAE is unsupervised learning, we do not need label
        return image_batch





