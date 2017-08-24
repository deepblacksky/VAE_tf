# coding=utf-8
from __future__ import print_function
from __future__ import division

import tensorflow as tf
import numpy as np


np.random.seed(0)
tf.set_random_seed(0)


class VariationalAutoEncoder(object):
    def __init__(self, data, activation_fct=tf.nn.relu,
                 n_hidden_encode_1=512, n_hidden_encode_2=512,
                 n_hidden_decode_1=512, n_hidden_decode_2=512,
                 n_input=28*28, n_z=10,
                 learning_rate=None, batch_size=None):
        self.data = data
        self.activation_fct = activation_fct
        self.n_hidden_encode_1 = n_hidden_encode_1
        self.n_hidden_encode_2 = n_hidden_encode_2
        self.n_hidden_decode_1 = n_hidden_decode_1
        self.n_hidden_decode_2 = n_hidden_decode_2
        self.n_input = n_input
        self.n_z = n_z
        self.learning_rate = learning_rate
        self.batch_size = batch_size

        self.z_mean = None
        self.log_z_variance_sq = None
        self.z = None
        self.x_reconstruct_mean = None
        self.x_reconstruct_mean_sigmoid = None
        self.losses = None
        self.optimizer = None

    def create_vae_network(self):
        self.z_mean, self.log_z_variance_sq = self._encode_network()

        # generator z
        # z = mu + variance * epsilon
        eps = tf.random_normal((self.batch_size, self.n_z), mean=0.0, stddev=1.0, dtype=tf.float32)
        with tf.variable_scope('z'):
            self.z = tf.add(self.z_mean, tf.multiply(tf.sqrt(tf.exp(self.log_z_variance_sq)), eps))
            self._summary_helper(self.z)
        self.x_reconstruct_mean, self.x_reconstruct_mean_sigmoid = self._decode_network()

    def create_loss_optimizer(self):
        # vae loss has two part, reference: https://arxiv.org/abs/1312.6114
        with tf.variable_scope('losses_and_optimizer'):
            # reconstr_loss = -tf.reduce_sum(self.data * tf.log(1e-10 + self.x_reconstruct_mean) +
            #                                (1 - self.data) * tf.log(1e-10 + (1 - self.x_reconstruct_mean)), 1)
            reconstr_loss = tf.reduce_sum(
                tf.nn.sigmoid_cross_entropy_with_logits(labels=self.data, logits=self.x_reconstruct_mean), 1)
            # KL loss
            latent_loss = -0.5 * tf.reduce_sum(1 + self.log_z_variance_sq -
                                               tf.square(self.z_mean) - tf.exp(self.log_z_variance_sq), 1)
            self.losses = tf.reduce_mean(reconstr_loss + latent_loss)
            tf.summary.scalar('total_losses', self.losses)
            self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.losses)

    def _encode_network(self):
        with tf.variable_scope('encode_network_layer_1'):
            encode_weights_h1 = tf.get_variable(name='encode_weights_h1',
                                                shape=[self.n_input, self.n_hidden_encode_1],
                                                dtype=tf.float32,
                                                initializer=tf.contrib.layers.xavier_initializer())
            encode_biases_h1 = tf.get_variable(name='encode_biases_h1',
                                               shape=[self.n_hidden_encode_1],
                                               dtype=tf.float32,
                                               initializer=tf.constant_initializer(0.1))
            tf.summary.histogram('encode_weights_h1', encode_weights_h1)
            tf.summary.histogram('encode_biases_h1', encode_biases_h1)
            encode_h1 = self.activation_fct(tf.matmul(self.data, encode_weights_h1) + encode_biases_h1,
                                            name='encode_network_layer_1')
            self._summary_helper(encode_h1)

        with tf.variable_scope('encode_network_layer_2'):
            encode_weights_h2 = tf.get_variable(name='encode_weights_h2',
                                                shape=[self.n_hidden_encode_1, self.n_hidden_encode_2],
                                                dtype=tf.float32,
                                                initializer=tf.contrib.layers.xavier_initializer())
            encode_biases_h2 = tf.get_variable(name='encode_biases_h2',
                                               shape=[self.n_hidden_encode_2],
                                               dtype=tf.float32,
                                               initializer=tf.constant_initializer(0.1))
            tf.summary.histogram('encode_weights_h2', encode_weights_h2)
            tf.summary.histogram('encode_biases_h2', encode_biases_h2)
            encode_h2 = self.activation_fct(tf.matmul(encode_h1, encode_weights_h2) + encode_biases_h2,
                                            name='encode_network_layer_2')
            self._summary_helper(encode_h2)

        with tf.variable_scope('encode_network_mean'):
            encode_weights_mean = tf.get_variable(name='encode_weights_mean',
                                                  shape=[self.n_hidden_encode_2, self.n_z],
                                                  dtype=tf.float32,
                                                  initializer=tf.contrib.layers.xavier_initializer())
            encode_biases_mean = tf.get_variable(name='encode_biases_mean',
                                                 shape=[self.n_z],
                                                 dtype=tf.float32,
                                                 initializer=tf.constant_initializer(0.1))
            tf.summary.histogram('encode_weights_mean', encode_weights_mean)
            tf.summary.histogram('encode_biases_mean', encode_biases_mean)
            encode_z_mean = tf.add(tf.matmul(encode_h2, encode_weights_mean), encode_biases_mean,
                                   name='encode_z_mean')
            self._summary_helper(encode_z_mean)

        with tf.variable_scope('encode_network_variance'):
            encode_weights_variance = tf.get_variable(name='encode_weights_variance',
                                                      shape=[self.n_hidden_encode_2, self.n_z],
                                                      dtype=tf.float32,
                                                      initializer=tf.contrib.layers.xavier_initializer())
            encode_biases_variance = tf.get_variable(name='encode_biases_variance',
                                                     shape=[self.n_z],
                                                     dtype=tf.float32,
                                                     initializer=tf.constant_initializer(0.1))
            tf.summary.histogram('encode_weights_variance', encode_weights_variance)
            tf.summary.histogram('encode_biases_variance', encode_biases_variance)
            # this z_variance is not real variance, it is log(sigma^2)
            encode_log_z_variance_sq = tf.add(tf.matmul(encode_h2, encode_weights_variance), encode_biases_variance,
                                              name='encode_z_variance')
            self._summary_helper(encode_log_z_variance_sq)

        return encode_z_mean, encode_log_z_variance_sq

    def _decode_network(self):
        with tf.variable_scope('decode_network_h1'):
            decode_weights_h1 = tf.get_variable(name='decode_weights_h1',
                                                shape=[self.n_z, self.n_hidden_decode_1],
                                                dtype=tf.float32,
                                                initializer=tf.contrib.layers.xavier_initializer())
            decode_biased_h1 = tf.get_variable(name='decode_biases_h1',
                                               shape=[self.n_hidden_decode_1],
                                               dtype=tf.float32,
                                               initializer=tf.constant_initializer(0.1))
            tf.summary.histogram('decode_weights_h1', decode_weights_h1)
            tf.summary.histogram('decode_biased_h1', decode_biased_h1)
            decode_h1 = self.activation_fct(tf.matmul(self.z, decode_weights_h1) + decode_biased_h1,
                                            name='decode_network_h1')
            self._summary_helper(decode_h1)

        with tf.variable_scope('decode_network_h2'):
            decode_weights_h2 = tf.get_variable(name='decode_weights_h2',
                                                shape=[self.n_hidden_decode_1, self.n_hidden_decode_2],
                                                dtype=tf.float32,
                                                initializer=tf.contrib.layers.xavier_initializer())
            decode_biased_h2 = tf.get_variable(name='decode_biases_h2',
                                               shape=[self.n_hidden_decode_2],
                                               dtype=tf.float32,
                                               initializer=tf.constant_initializer(0.1))
            tf.summary.histogram('decode_weights_h2', decode_weights_h2)
            tf.summary.histogram('decode_biased_h2', decode_biased_h2)
            decode_h2 = self.activation_fct(tf.matmul(decode_h1, decode_weights_h2) + decode_biased_h2,
                                            name='decode_network_h2')
            self._summary_helper(decode_h2)

        with tf.variable_scope('decode_network_x_reconstruct'):
            x_reconstruct_weights = tf.get_variable(name='decode_network_x_reconstruct_weights',
                                                    shape= [self.n_hidden_decode_2, self.n_input],
                                                    dtype=tf.float32,
                                                    initializer=tf.contrib.layers.xavier_initializer())
            x_reconstruct_biases = tf.get_variable(name='decode_network_x_reconstruct_biases',
                                                   shape=[self.n_input],
                                                   dtype=tf.float32,
                                                   initializer=tf.constant_initializer(0.1))
            tf.summary.histogram('x_reconstruct_weights', x_reconstruct_weights)
            tf.summary.histogram('x_reconstruct_biases', x_reconstruct_biases)
            x_reconstruct_mean = tf.add(tf.matmul(decode_h2, x_reconstruct_weights), x_reconstruct_biases)
            x_reconstruct_mean_sigmoid = tf.nn.sigmoid(x_reconstruct_mean, name='decode_network_x_reconstruct')
            self._summary_helper(x_reconstruct_mean_sigmoid)
        return x_reconstruct_mean, x_reconstruct_mean_sigmoid

    @staticmethod
    def _summary_helper(variable):
        tf.summary.histogram(variable.op.name + '/activation', variable)
        tf.summary.scalar(variable.op.name + '/mean', tf.reduce_mean(variable))
