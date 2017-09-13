import tensorflow as tf

a = tf.constant(10, name='a')
b = tf.constant(90, name='b')
y = tf.Variable(a+b*2, name='y')

model = tf.initialize_all_variables()
with tf.Session() as s:
	merged = tf.summary.merge_all()							# merges all collected summaries on the default graph
	writer = tf.summary.FileWriter('./', graph=s.graph)		# write it directly from TF session

	s.run(model)
	print(s.run(y))

# then to see on tensorboard
# $ tensorboard --logdir=.
