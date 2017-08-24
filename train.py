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
