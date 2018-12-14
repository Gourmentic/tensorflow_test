import tensorflow as tf
import numpy as np

'''

sess = tf.Session()

x_vals = np.random.normal(1,0.1,100)
y_vals = np.repeat(10,100)

x_data = tf.placeholder(dtype=tf.float32,shape=[1])
y_target = tf.placeholder(dtype=tf.float32,shape=[1])

A = tf.Variable(initial_value=np.random.normal(1))

my_output = tf.multiply(x_data,A)

init = tf.initialize_all_variables()
sess.run(init)

loss = tf.square(y_target-my_output)

my_opt = tf.train.GradientDescentOptimizer(learning_rate=0.02)
train_step = my_opt.minimize(loss)


for i in range(200):

    rand_index = np.random.choice(100)
    rand_x = [x_vals[rand_index]]
    rand_y = [y_vals[rand_index]]

    sess.run(train_step,feed_dict={x_data:rand_x,y_target:rand_y})

    if (i+1)%5 ==0:
        print('Step #'+ str(i+1) +' A = '+ str(sess.run(A)))
        print('Loss = ' + str(sess.run(loss, feed_dict={x_data:rand_x, y_target: rand_y})))

from tensorflow.python.framework import ops
ops.reset_default_graph()


sess = tf.Session()

x_vals = np.concatenate(
    [np.random.normal(-1,1,(50,)),np.random.normal(3,1,(50,))])

y_vals = np.concatenate(
    [np.repeat(0.,50),np.repeat(1.,50)])

x_data = tf.placeholder(shape=[1],dtype=tf.float32)
y_target = tf.placeholder(shape=[1],dtype=tf.float32)

A = tf.Variable(tf.random_normal(shape=[1],mean=10.))

my_output = tf.add(x_data,A)

my_output_expanded = tf.expand_dims(my_output,0)
y_target_expanded = tf.expand_dims(y_target,0)

init = tf.initialize_all_variables()
sess.run(init)

xentropy =\
    tf.nn.sigmoid_cross_entropy_with_logits(
        logits=my_output_expanded,
        labels=y_target_expanded)

my_opt = tf.train.GradientDescentOptimizer(learning_rate=0.01)
train_step = my_opt.minimize(xentropy)

for i in range(1400):

    rand_index = np.random.choice(100)
    rand_x = [x_vals[rand_index]]
    rand_y = [y_vals[rand_index]]

    sess.run(train_step, feed_dict={x_data:rand_x,y_target:rand_y})

    if (i+1)%100 == 0:

        print('Step #'+ str(i+1) + ' A = ' + str(sess.run(A)))
        print('Loss = ' + str(sess.run(xentropy,feed_dict={x_data:rand_x,y_target:rand_y})))

'''
sess = tf.Session()


x_vals = np.random.normal(1,0.1,100).reshape(-1,1)
y_vals = np.repeat(10,100).reshape(-1,1)

x_data = tf.placeholder(dtype=tf.float32,shape=[None,1])
y_target = tf.placeholder(dtype=tf.float32,shape=[None,1])

A = tf.Variable(initial_value=[[1.]])

init = tf.global_variables_initializer()
sess.run(init)

my_output = tf.matmul(x_data,A)

loss = tf.reduce_max(tf.square(my_output-y_target))

my_opt = tf.train.GradientDescentOptimizer(learning_rate=0.01)
train_step = my_opt.minimize(loss)

loss_batch = []
batch_size = 15
data_size = len(x_vals)

for epoch in range(20):
    for i in range((data_size//batch_size)+1):

        rand_x = x_vals[i*batch_size:(i+1)*batch_size]
        rand_y = y_vals[i*batch_size:(i+1)*batch_size]

        sess.run(train_step,feed_dict={x_data:rand_x,y_target:rand_y})


    print('epoch #' + str(epoch+1) + ' A = '+str(sess.run(A)))
    temp_loss = sess.run(loss,feed_dict={x_data: rand_x,y_target:rand_y})
    print('Loss = ' +str(temp_loss))

    loss_batch.append(temp_loss)
    print(loss_batch)


sess.run(init)
#for i in range(1000):
loss_stochastic = []



data_list = list(range(data_size))
np.random.shuffle(data_list)

for i,rand_index in enumerate(data_list):
    rand_x = x_vals[rand_index].reshape(-1,1)
    rand_y = y_vals[rand_index].reshape(-1,1)

    sess.run(train_step,feed_dict={x_data:rand_x,y_target:rand_y})

    if i%5 == 0:
        print('step #' + str(i+1) + ' A = '+str(sess.run(A)))
        temp_loss = sess.run(loss,feed_dict={x_data: rand_x,y_target:rand_y})
        print('Loss = ' +str(temp_loss))

        loss_stochastic.append(temp_loss)
        print(loss_stochastic)
#
import matplotlib.pyplot as plt

plt.plot(range(20),loss_batch,'b--',label='Batch Loss(size=15)')
plt.plot(range(20),loss_stochastic,'r-',label='Stochastic Loss')
plt.legend(loc='upper right', prop={'size':11})
plt.show()

