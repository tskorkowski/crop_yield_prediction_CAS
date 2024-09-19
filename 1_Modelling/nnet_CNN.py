""" Code modified from https://github.com/JiaxuanYou/crop_yield_prediction"""

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from datetime import datetime
import math

# Disable eager execution to enable TensorFlow 1.x style operations in TensorFlow 2.x
tf.compat.v1.disable_eager_execution()

class CNN_Config():
    # there are 32 buckets in the histogram, 32 timesteps per year, and 9 "bands"
    season_len = 27
    B, W, H, C = 64, 32, 27, 9
    
    loss_lambda = 0.5
    train_step = 10000000
    lr = 1e-4
    weight_decay = 0.005
    drop_out = 0.5

    def __init__(self, season_frac=None):
        if season_frac is not None:
            self.H = int(season_frac*self.H)
    
def conv2d(input_data, out_channels, filter_size, stride, in_channels=None, name="conv2d"):
    if not in_channels:
        in_channels = input_data.get_shape()[-1]
    with tf.compat.v1.variable_scope(name):
        W = tf.compat.v1.get_variable("W", [filter_size, filter_size, in_channels, out_channels],
                                      initializer=tf.compat.v1.keras.initializers.he_normal())
        b = tf.compat.v1.get_variable("b", [out_channels], initializer=tf.zeros_initializer())
        tf.compat.v1.summary.histogram(name + ".W", W)
        tf.compat.v1.summary.histogram(name + ".b", b)
        return tf.nn.conv2d(input_data, W, strides=[1, stride, stride, 1], padding="SAME") + b

def pool2d(input_data, ksize, name="pool2d"):
    with tf.variable_scope(name):
        return tf.nn.max_pool(input_data, [1, ksize, ksize, 1], [1, ksize, ksize, 1], "SAME")

def conv_relu_batch(input_data, out_channels, filter_size, stride, in_channels=None, name="crb"):
    with tf.compat.v1.variable_scope(name):
        a = conv2d(input_data, out_channels, filter_size, stride, in_channels)
        tf.summary.histogram(name + ".a", a)
        b = batch_normalization(a, axes=[0, 1, 2])
        r = tf.nn.relu(b)
        tf.summary.histogram(name + ".r", r)
        return r

def dense(input_data, H, N=None, name="dense"):
    if not N:
        N = input_data.get_shape()[-1]
    with tf.compat.v1.variable_scope(name):
        W = tf.compat.v1.get_variable("W", [N, H], initializer=tf.keras.initializers.he_normal())
        b = tf.compat.v1.get_variable("b", [H], initializer=tf.zeros_initializer())
        tf.summary.histogram(name + ".W", W)
        tf.summary.histogram(name + ".b", b)
        return tf.matmul(input_data, W, name="matmul") + b

def batch_normalization(input_data, axes=[0], name="batch"):
    with tf.compat.v1.variable_scope(name):
        mean, variance = tf.nn.moments(input_data, axes, keepdims=True, name="moments")
        tf.summary.histogram(name + ".mean", mean)
        tf.summary.histogram(name + ".variance", variance)
        return tf.nn.batch_normalization(input_data, mean, variance, None, None, 1e-6, name="batch")

class CNN_NeuralModel():
    def __init__(self, config, name):
        self.x = tf.compat.v1.placeholder(tf.float32, [None, config.W, config.H, config.C], name="x")
        self.y = tf.compat.v1.placeholder(tf.float32, [None])
        self.lr = tf.compat.v1.placeholder(tf.float32, [])
        self.keep_prob = tf.compat.v1.placeholder(tf.float32, [])

        # Normalize input data
        self.x_norm = tf.divide(self.x, 255.0)  # Assuming the input data range is [0, 255]

        self.conv1_1 = conv_relu_batch(self.x_norm, 64, 3, 1, name="conv1_1")
        conv1_1_d = tf.nn.dropout(self.conv1_1, rate=1 - self.keep_prob)
        self.conv1_2 = conv_relu_batch(conv1_1_d, 64, 3, 2, name="conv1_2")
        conv1_2_d = tf.nn.dropout(self.conv1_2, rate=1 - self.keep_prob)

        self.conv2_1 = conv_relu_batch(conv1_2_d, 128, 3, 1, name="conv2_1")
        conv2_1_d = tf.nn.dropout(self.conv2_1, rate=1 - self.keep_prob)
        self.conv2_2 = conv_relu_batch(conv2_1_d, 128, 3, 2, name="conv2_2")
        conv2_2_d = tf.nn.dropout(self.conv2_2, rate=1 - self.keep_prob)

        self.conv3_1 = conv_relu_batch(conv2_2_d, 256, 3, 1, name="conv3_1")
        conv3_1_d = tf.nn.dropout(self.conv3_1, rate=1 - self.keep_prob)
        self.conv3_2 = conv_relu_batch(conv3_1_d, 256, 3, 1, name="conv3_2")
        conv3_2_d = tf.nn.dropout(self.conv3_2, rate=1 - self.keep_prob)
        self.conv3_3 = conv_relu_batch(conv3_2_d, 256, 3, 2, name="conv3_3")
        conv3_3_d = tf.nn.dropout(self.conv3_3, rate=1 - self.keep_prob)

        dim = np.prod(conv3_3_d.get_shape().as_list()[1:])
        flattened = tf.reshape(conv3_3_d, [-1, dim])

        self.feature = dense(flattened, 1024, name="feature")

        self.pred = tf.squeeze(dense(self.feature, 1, name="dense"))
        self.loss_err = tf.nn.l2_loss(self.pred - self.y)

        self.loss_reg = tf.add_n([tf.nn.l2_loss(v) for v in tf.compat.v1.trainable_variables()])
        self.loss = config.loss_lambda * self.loss_err + (1 - config.loss_lambda) * self.loss_reg

        loss_err_summary_op = tf.compat.v1.summary.scalar("CNN/loss_err", self.loss_err)
        loss_reg_summary_op = tf.compat.v1.summary.scalar("CNN/loss_reg", self.loss_reg)
        loss_total_summary_op = tf.compat.v1.summary.scalar("CNN/loss", self.loss)
        self.loss_summary_op = tf.compat.v1.summary.merge([loss_err_summary_op, loss_reg_summary_op, loss_total_summary_op])

        self.train_op = tf.compat.v1.train.AdamOptimizer(self.lr).minimize(self.loss)
        self.summary_op = None

        self.writer = None
        self.saver = tf.compat.v1.train.Saver()

        with tf.compat.v1.variable_scope('dense', reuse=tf.compat.v1.AUTO_REUSE):
            self.dense_W = tf.compat.v1.get_variable('W')
            self.dense_B = tf.compat.v1.get_variable('b')


# Additional training loop function
def train_model():
    config = CNN_Config()
    model = CNN_NeuralModel(config, name="CNN_Model")
    
    with tf.compat.v1.Session() as sess:
        sess.run(tf.compat.v1.global_variables_initializer())
        
        # Assuming you have training data `X_train` and `y_train`
        # Adjust the following lines to load your actual data
        X_train = np.random.rand(100, config.W, config.H, config.C)  # Dummy data
        y_train = np.random.rand(100)  # Dummy labels
        
        feed_dict = {model.x: X_train, model.y: y_train, model.lr: config.lr, model.keep_prob: config.drop_out}
        
        for step in range(config.train_step):
            _, loss, summary = sess.run([model.train_op, model.loss, model.loss_summary_op], feed_dict=feed_dict)
            
            if step % 100 == 0:
                print(f'Step {step}, Loss: {loss}')
                
                # Add summary to TensorBoard
                if model.writer is None:
                    model.writer = tf.compat.v1.summary.FileWriter('logs', sess.graph)
                model.writer.add_summary(summary, step)
        
        # Save the trained model
        model.saver.save(sess, 'model.ckpt')

if __name__ == "__main__":
    train_model()
