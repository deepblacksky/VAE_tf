# coding=utf-8
from __future__ import print_function
from __future__ import division

import tensorflow as tf
import numpy as np
import os
import time

import readData
import vae_model

np.random.seed(0)
tf.set_random_seed(0)


FLAGS = tf.flags.FLAGS

tf.flags.DEFINE_string('buckets', 'D:\\DL\\aliyun\\VAE\\data', '数据目录')
tf.flags.DEFINE_string('checkpointDir', 'D:\DL\\aliyun\\VAE\\model', '模型保存路径')
tf.flags.DEFINE_string('summaryDir', 'D:\\DL\\aliyun\\VAE\\logs', 'tensorboard保存路径')
tf.flags.DEFINE_integer('batch_size', 256, 'Batch Size')
tf.flags.DEFINE_float('learning_rate', 1e-3, 'learning rate')
tf.flags.DEFINE_integer('display_step', 500, 'Display step')
tf.flags.DEFINE_float('train_step', 100000, 'Train step')

train_file_path = os.path.join(FLAGS.buckets, 'train.tfrecords')

# read data
reader = readData.MnistReader(train_file_path, batch_size=FLAGS.batch_size)
image_batch = reader.read_image_batch()

x = tf.placeholder(dtype=tf.float32, shape=[FLAGS.batch_size, 28*28])
vae = vae_model.VariationalAutoEncoder(data=x,
                                       learning_rate=FLAGS.learning_rate,
                                       batch_size=FLAGS.batch_size)

vae.create_vae_network()
print('create VAE model network')

vae.create_loss_optimizer()

sess = tf.InteractiveSession()
summary = tf.summary.FileWriter(FLAGS.summaryDir, graph=sess.graph)
saver = tf.train.Saver(var_list=tf.trainable_variables())
tf.global_variables_initializer().run()
tf.local_variables_initializer().run()
coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(sess=sess, coord=coord)

try:
    for step in range(FLAGS.train_step):
        if coord.should_stop():
            break
        train_image_batch = sess.run(image_batch)
        _, losses = sess.run(fetches=[vae.optimizer, vae.losses], feed_dict={x: train_image_batch})

        if step % FLAGS.display_step == 0:
            summary_data = tf.summary.merge_all()
            summary.add_summary(sess.run(summary_data, feed_dict={x: train_image_batch}), step)
            print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
            print('Step:', '%05d' % step, 'losses:', '{:.8f}'.format(losses))

except tf.errors.OutOfRangeError:
    print('train done')
finally:
    coord.request_stop()

saver.save(sess=sess, save_path=os.path.join(FLAGS.checkpointDir, 'vae.model'))

coord.join(threads)
sess.close()
summary.close()
print('train done!')
