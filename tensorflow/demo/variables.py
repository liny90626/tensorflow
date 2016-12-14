import tensorflow as tf
state = tf.Variable(0, name="test")
one = tf.constant(1)

new_value = tf.add(state, one);
update = tf.assign(state, new_value);

init_op = tf.global_variables_initializer()

with tf.Session() as sess:
	#state.initializer.run()
	sess.run(init_op)
	
	print(sess.run(update))
	
	for _ in range(3):
		result = sess.run(update)
		print(result)
