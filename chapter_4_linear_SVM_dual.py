import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import StratifiedShuffleSplit

digits_data = datasets.load_digits()
feature_data = digits_data['data']
raw_label_data = digits_data['target']
label_table = np.zeros(shape=(len(label_data), 10))

for i in range(len(label_data)):

    label_table[i, label_data[i]] = 1


sess = tf.Session()

# split the train and test dataset
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.3)

for train_index, test_index in sss.split(feature_data, label_data):

    train_x, train_y = feature_data[train_index], label_table[train_index]
    test_x, test_y = feature_data[test_index], label_table[test_index]

train_size = train_x.shape[0]
test_size = test_x.shape[0]

# stochastic batch training can not be used, the whole data should be
# trained together

X_train_input = tf.placeholder(shape=(train_size, 64), dtype=tf.float32)
Y_target = tf.placeholder(shape=(train_size, 1), dtype=tf.float32)
alpha_weight = tf.Variable(np.random.rand(train_size, 1), dtype=tf.float32)

# ensure the KKT
alpha_weight = tf.maximun(alpha_weight, 0)
gamma = tf.constant([0.5], dtype=tf.float32)


weight_term = tf.reduce_sum(alpha_weight)

x1 = tf.expand_dims(X_train_input, axis=1)
x2 = tf.expand_dims(X_train_input, axis=0)
x1 = tf.broadcast_to(x1, shape=(train_size, train_size, 64))
x2 = tf.broadcast_to(x2, shape=(train_size, train_size, 64))

# RBF kernel k<xi,xj> = exp(-gamma*(xi-xj)**2)
kernel = tf.exp(tf.reduce_sum((x1 - x2)**2, axis=2))

Y_alpha = tf.multiply(Y_target, alpha_weight)
Y_alpha_cross = tf.matmul(Y_alpha, tf.transpose(Y_alpha))

kernel_term = tf.reduce_sum(tf.multiply(Y_alpha_cross, kernel))

loss = 0.5 * kernel_term - weight_term

my_opt = tf.train.GradientDescentOptimizer(learing_rate=0.1)

train_step = my_opt.minimize(loss)

init = tf.global_variables_initializer()
init.run()

for epoch in range(10):

    sess.run(train_step,feed_dict={X_train_input:train_x,Y_target:train_y})
    alpha_weight_cal = sess.run()
    print()



#get the support vector



X_test_input = tf.placeholder(shape=(test_size, 64), dtype=tf.float32)
X_support_input = tf.placeholder(shape=(train_size, 64), dtype=tf.float32)
Y_support_input = tf.placeholder(shape=(train_size, 1), dtype=tf.float32)
alpha_weight_cal = sess.run(alpha_weight)

x1 = tf.expand_dims(X_train_input, axis=1)
x2 = tf.expand_dims(X_train_input, axis=0)
x1 = tf.broadcast_to(x1, shape=(test_size, train_size, 64))
x2 = tf.broadcast_to(x2, shape=(test_size, train_size, 64))

kernel = tf.exp(tf.reduce_sum((x1 - x2)**2, axis=2))

bais = tf.constant([10.])

weight_Y_cal = tf.multiply(Y_support_input, alpha_weight_cal)



# Y_test_target_prd =
#


# the instance should be created throught dual problem
# one vs others will be used for mutli-classification
