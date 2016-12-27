import tensorflow as tf
# 1x2
matrix1 = tf.constant([[2., 3.], [4., 5.]])
# 2x1
matrix2 = tf.constant([[1.], [2.]])

# This is a graph, represtens computations
product = tf.matmul(matrix1, matrix2)

# This is a session, used to execute graphs
sess = tf.Session()

result = sess.run(product)

print(result)

# Close to release resources
sess.close()
