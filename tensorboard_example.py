import tensorflow as tf
import numpy as np
import os
import io
import time
import matplotlib.pyplot as plt

sess = tf.Session()
epochs = 20
lr = 0.0004

if not os.path.exists('tensorboard'):
    os.mkdir('tensorboard')


raw_data = np.linspace(0., 10., 100)
# x_data = np.c_[raw_data, np.square(raw_data), np.power(raw_data, 3)]
# y_data = 10 * raw_data + 2 * \
#     np.square(raw_data) + np.power(raw_data, 3) + 20 + np.random.rand(100)

x_data = np.c_[raw_data, np.square(raw_data)]
y_data = 10 * raw_data + 2 * np.square(raw_data) + np.random.rand(100)


y_data = y_data.reshape(-1, 1)

X_input = tf.placeholder(shape=(None, 2), dtype=tf.float32)
Y_input = tf.placeholder(shape=(None, 1), dtype=tf.float32)
weight = tf.Variable(
    initial_value=tf.random_uniform(
        shape=(
            2,
            1)),
    dtype=tf.float32)
bias = tf.Variable(initial_value=np.array([[0.]]), dtype=tf.float32)

# tf.add or +
# model = tf.matmul(X_input,weight) + bias
model = tf.add(tf.matmul(X_input, weight), bias)
loss = tf.reduce_mean(tf.square(tf.subtract(model, Y_input)))
train_step = tf.train.GradientDescentOptimizer(learning_rate=lr).minimize(loss)


with tf.name_scope('Weight_Estimate'):
    tf.summary.scalar('bias', tf.squeeze(bias))
    tf.summary.histogram('slope',tf.squeeze(weight))

with tf.name_scope('Loss_Estimate'):
    tf.summary.scalar('L2_loss', loss)

summary_op = tf.summary.merge_all()
summary_weight_op = tf.summary.merge(['bias'])

init = tf.global_variables_initializer()
sess.run(init)

summary_writer = tf.summary.FileWriter(logdir='tensorboard', graph=sess.graph)


for epoch in range(epochs):

    train_index = np.random.choice(100,10)
    x_train = x_data[train_index]
    y_train = y_data[train_index]

    _, train_loss, summary = sess.run([train_step, loss, summary_op], feed_dict={
                                      X_input: x_data, Y_input: y_data})

    summary_writer.add_summary(summary,epoch)
    print('at {}, the training loss: {:.2f}'.format(epoch, train_loss))


    y_prd = sess.run(model,feed_dict={X_input: x_data})

    def gen_linear_plot(x_data,y_data,y_prd):
        plt.plot(x_data,y_data,'--r',label='True value')
        plt.plot(x_data,y_prd,'-g',label='predict value')
        plt.legend()
        buf = io.BytesIO()
        plt.savefig(buf,format='png')
        buf.seek(0)

        return buf

    plot_buf = gen_linear_plot(x_data,y_prd,y_prd)
    image = tf.image.decode_png(plot_buf.getvalue(),channels=4)
    print(image.shape)
    image = tf.expand_dims(image,0)
    image_summary_op = tf.summary.image('Linear plot',image)
    image_summary = sess.run(image_summary_op)
    summary_writer.add_summary(image_summary, epoch)

summary_writer.close()