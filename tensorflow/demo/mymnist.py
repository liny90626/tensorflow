#coding:utf-8
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

import tensorflow as tf

######################## 建模
x = tf.placeholder("float", [None, 784])

# 使用变量来维护状态
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

# 构建模型
y = tf.nn.softmax(tf.matmul(x, W) + b)

# 交叉熵(衡量成本/开销)
yr = tf.placeholder("float", [None, 10])
crossEntropy = -tf.reduce_sum(yr*tf.log(y))
#crossEntropy = tf.reduce_mean(tf.square(yr - y))

# 训练方法
trainStep = tf.train.GradientDescentOptimizer(0.01).minimize(crossEntropy)

# 测试方法
correctPrediction = tf.equal(tf.argmax(y, 1), tf.argmax(yr, 1))
accuracy = tf.reduce_mean(tf.cast(correctPrediction, "float"))

######################## 训练
init = tf.global_variables_initializer()
sess = tf.Session();
sess.run(init)

for i in range(1000):
	batch_xs, batch_ys = mnist.train.next_batch(100)
	sess.run(trainStep, feed_dict={x: batch_xs, yr: batch_ys})

print sess.run(accuracy, feed_dict={x: mnist.test.images, yr: mnist.test.labels})
	
sess.close()