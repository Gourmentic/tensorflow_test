import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from matplotlib import animation

# data_prepare
iris_data = datasets.load_iris()

y_data = iris_data['target'][iris_data['target'] != 2]
x_data = iris_data['data'][iris_data['target'] != 2, 1:3]

y_data = np.where(y_data == 0, 1, -1)

x_train, x_test, y_train, y_test = \
    train_test_split(
        x_data,
        y_data,
        test_size=0.33,
        random_state=42,
        shuffle=True)

y_train = y_train.astype(np.float32).reshape(-1, 1)
y_test = y_test.astype(np.float32).reshape(-1, 1)


train_size = x_train.shape[0]
test_size = x_test.shape[0]
feature_dim = 2


sess = tf.Session()
# stochastic batch training can not be used, the whole data should be
# trained together

X_train_input = tf.placeholder(
    shape=(
        train_size,
        feature_dim),
    dtype=tf.float32)
Y_target = tf.placeholder(shape=(train_size, 1), dtype=tf.float32)
alpha_weight = tf.Variable(
    np.random.rand(
        train_size,
        1) / 10,
    dtype=tf.float32)

# ensure the KKT
alpha_weight = tf.maximum(alpha_weight, 0)
gamma = tf.constant([0.5], dtype=tf.float32)


weight_term = tf.reduce_sum(alpha_weight)

x1 = tf.expand_dims(X_train_input, axis=1)
x2 = tf.expand_dims(X_train_input, axis=0)
x1 = tf.broadcast_to(x1, shape=(train_size, train_size, feature_dim))
x2 = tf.broadcast_to(x2, shape=(train_size, train_size, feature_dim))

# RBF kernel k<xi,xj> = exp(-gamma*(xi-xj)**2)
#kernel = tf.exp(tf.reduce_sum((x1 - x2)**2, axis=2))

# linear kernel K<xi,xj> = xi dot xj
kernel = tf.reduce_sum(tf.multiply(x1, x2), axis=2)

Y_alpha = tf.multiply(Y_target, alpha_weight)
Y_alpha_cross = tf.matmul(Y_alpha, tf.transpose(Y_alpha))

kernel_term = tf.reduce_sum(tf.multiply(Y_alpha_cross, kernel))

loss = tf.subtract(tf.multiply(kernel_term, 0.5), weight_term)

my_opt = tf.train.GradientDescentOptimizer(learning_rate=0.001)

train_step = my_opt.minimize(loss)

init = tf.global_variables_initializer()
sess.run(init)

alpha_weight_list = []

for epoch in range(5000):

    sess.run(train_step, feed_dict={X_train_input: x_train, Y_target: y_train})
    alpha_weight_cal = sess.run(alpha_weight)
    loss_cal = sess.run(
        loss,
        feed_dict={
            X_train_input: x_train,
            Y_target: y_train})

    # print(alpha_weight_cal)
    alpha_weight_list.append(alpha_weight_cal)

    print('When epoch %d' % epoch, loss_cal)


plt.plot(alpha_weight_cal, '*b', label='alpha_weight')
plt.title('Weight @')
plt.show()

#get the support vector
support_select = (alpha_weight_cal>0.05).reshape(-1)
support_x = x_train[support_select]
support_y = y_train[support_select]
support_weight = alpha_weight_cal[support_select]

# calculate the weight and bias

weight_cal = np.sum(alpha_weight_cal * y_train * x_train, axis=0)

support_kernel = np.sum(np.matmul(
    support_x, np.transpose(support_x)), axis=1).reshape(-1,1)

b_cal = np.mean(1/support_y-(support_weight*support_y*support_kernel))


x1 = np.linspace(2, 4.5, 2)
x2 = (-(weight_cal[0] * x1 + b_cal) / weight_cal[1]).reshape(-1)

train_1 = (y_train == 1).reshape(-1)
train_0 = (y_train == -1).reshape(-1)
plt.plot(x_train[train_1, 0], x_train[train_1, 1], '*r')
plt.plot(x_train[train_0, 0], x_train[train_0, 1], '*b')
plt.plot(support_x[:,0], support_x[:,1], 'oy')
plt.plot(x1, x2, '-')
plt.show()

print(support_x)
# get the support vector


# X_test_input = tf.placeholder(shape=(test_size, 64), dtype=tf.float32)
# X_support_input = tf.placeholder(shape=(train_size, 64), dtype=tf.float32)
# Y_support_input = tf.placeholder(shape=(train_size, 1), dtype=tf.float32)
# alpha_weight_cal = sess.run(alpha_weight)
#
# x1 = tf.expand_dims(X_train_input, axis=1)
# x2 = tf.expand_dims(X_train_input, axis=0)
# x1 = tf.broadcast_to(x1, shape=(test_size, train_size, 64))
# x2 = tf.broadcast_to(x2, shape=(test_size, train_size, 64))
#
# kernel = tf.exp(tf.reduce_sum((x1 - x2)**2, axis=2))
#
# bais = tf.constant([10.])
#
# weight_Y_cal = tf.multiply(Y_support_input, alpha_weight_cal)
#

# Y_test_target_prd =
#


# the instance should be created throught dual problem
# one vs others will be used for mutli-classification
