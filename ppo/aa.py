import numpy as np
import os
import time
import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions

config = tf.ConfigProto()
os.environ["CUDA_VISIBLE_DEVICES"] = '1'
config.log_device_placement = False
config.gpu_options.allow_growth = True

with tf.Session(config=config) as sess:

    #input = np.asarray([[1, 1], [1, 1]])

    #state = tf.get_variable(input)  # 2 * 2

    #state = tf.Variable([[1., 1.], [1., 1.]])

    state = tf.placeholder(tf.float32, [2, 2])

    a = tf.layers.dense(state, 2, activation=tf.nn.sigmoid)

    output = tf.contrib.distributions.Normal(tf.zeros(shape=(2, 2)), a)
    output2 = tfd.Normal(loc=tf.zeros(shape=(2, 2)), scale=a)

    #output = tfd.Normal(loc=[0., 0.], scale=a)

    sess.run(tf.global_variables_initializer())

    print(output.batch_shape)

    print(sess.run(a, feed_dict={state: [[1., 1.], [1., 1.]]}))

    print(sess.run(output.prob([100., 0.]), feed_dict={state: [[1., 1.], [1., 1.]]}))

    t1 = time.time()
    print(sess.run(output.sample([1000]), feed_dict={state: [[1., 1.], [1., 1.]]}))
    t2 = time.time()

    print(sess.run(output2.sample([1000]), feed_dict={state: [[1., 1.], [1., 1.]]}))
    t3 = time.time()

    print(t2-t1, t3-t2)


'''
    output = tfd.Normal(loc=[[0, 0], [1, 1]], scale=[[1, 2], [1, 2]])

    print(output.batch_shape)

    print(sess.run(output.prob([[0, 0], [0, 0]])))

    print(sess.run(output.sample(2)))
'''