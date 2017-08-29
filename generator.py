# coding=utf-8
from __future__ import print_function
from __future__ import division

import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt

import vae_model

np.random.seed(0)
tf.set_random_seed(0)


FLAGS = tf.flags.FLAGS

tf.flags.DEFINE_string('buckets', 'D:\\DL\\aliyun\\VAE\\data', '数据目录')
tf.flags.DEFINE_string('checkpointDir', 'D:\DL\\aliyun\\VAE\\model', '模型保存路径')
tf.flags.DEFINE_string('summaryDir', 'D:\\DL\\aliyun\\VAE\\logs', 'tensorboard保存路径')
tf.flags.DEFINE_integer('batch_size', 1, 'Batch Size')
tf.flags.DEFINE_float('learning_rate', 1e-3, 'learning rate')

x = tf.placeholder(dtype=tf.float32, shape=[FLAGS.batch_size, 28*28])
vae = vae_model.VariationalAutoEncoder(data=x,
                                       learning_rate=FLAGS.learning_rate,
                                       batch_size=FLAGS.batch_size)
vae.create_vae_network()

sess = tf.InteractiveSession()
summary = tf.summary.FileWriter(FLAGS.summaryDir, graph=sess.graph)
saver = tf.train.Saver(var_list=tf.trainable_variables())
saver.restore(sess=sess, save_path=os.path.join(FLAGS.checkpointDir, 'vae.model'))

# generate image
nx = ny = 20
x_values = np.linspace(-2, 2, nx)
y_values = np.linspace(-2, 2, ny)

canvas = np.empty((28*nx, 28*ny))
for i, yi in enumerate(x_values):
    for j, xi in enumerate(y_values):
        z_mu = np.array([[xi, yi, xi, yi, xi, yi, xi, yi, xi, yi]]*FLAGS.batch_size)
        x_mean = sess.run(vae.x_reconstruct_mean_sigmoid, feed_dict={vae.z: z_mu})
        canvas[(nx-1-i)*28:(nx-i)*28, j*28:(j+1)*28] = x_mean.reshape(28, 28)

batch_xs = canvas*255
batch_xs = np.asarray(batch_xs, dtype=np.uint8)
# plt.imshow(batch_xs, cmap=plt.cm.gray)
plt.imsave('result.png', batch_xs, cmap=plt.cm.gray)
image_summary = tf.summary.image('gen_image', tf.expand_dims(tf.expand_dims(batch_xs, 0), 3))

summary_data = tf.summary.merge_all()
summary.add_summary(sess.run(image_summary), 0)
summary.close()
print("图像生成成功")
