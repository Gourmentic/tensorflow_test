import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets
import tensorflow as tf
from sklearn.model_selection import train_test_split

sess = tf.Session()

iris_data = datasets.load_iris()

y_data = iris_data['target'][iris_data['target']!=2]
x_data = iris_data['data'][iris_data['target']!=2,1:3]


x_train,x_test,y_train,y_test = \
    train_test_split(
        x_data,
        y_data,
        test_size=0.33,
        random_state=42,
        shuffle=True)

y_train = y_train.astype(np.float32).reshape(-1,1)
y_test = y_test.astype(np.float32).reshape(-1,1)

X_data = tf.placeholder(dtype=tf.float32,shape=(None,2))
Y_target = tf.placeholder(dtype=tf.float32,shape=(None,1))

w = tf.Variable(initial_value=tf.random_normal(shape=(2,1)))
b = tf.Variable(initial_value=tf.random_normal(shape=(1,1)))

init = tf.global_variables_initializer()

sess.run(init)


Y_logit = tf.add(tf.matmul(X_data,w),b)
Y_prediction = tf.round(tf.nn.sigmoid(Y_logit))

correct_prediction = tf.equal(Y_prediction,Y_target)

accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=Y_logit,labels=Y_target))

my_opt = tf.train.GradientDescentOptimizer(learning_rate=0.02)

train_step = my_opt.minimize(loss)


batch_size = 20
train_size = len(x_train)
epoch_train_loss = []
epoch_test_loss = []
epoches_size = 30

for epoch in range(epoches_size):

    for i in range((train_size//batch_size)+1):

        x_vals = x_train[i*batch_size:(i+1)*batch_size]
        y_vals = y_train[i*batch_size:(i+1)*batch_size]

        sess.run(train_step,feed_dict={X_data:x_vals,Y_target:y_vals})

    print('epoch #',epoch)
    train_loss = sess.run(loss,feed_dict={X_data:x_train,Y_target:y_train})
    test_loss = sess.run(loss,feed_dict={X_data:x_test,Y_target:y_test})
    train_acc = sess.run(accuracy,feed_dict={X_data:x_train,Y_target:y_train})
    test_acc = sess.run(accuracy,feed_dict={X_data:x_test,Y_target:y_test})
    print('the training loss:{:2f}'.format(train_loss))
    print('the test loss:{:2f}'.format(test_loss))
    print('the training acc: {:2f}'.format(train_acc))
    print('the testing acc: {:2f}'.format(test_acc))
    print('parameters:')
    print(sess.run(w),',',sess.run(b))
    epoch_train_loss.append(train_loss)
    epoch_test_loss.append(test_loss)

plt.plot(range(epoches_size),epoch_train_loss,'b-',label='Training loss')
plt.plot(range(epoches_size),epoch_test_loss,'r--',label='Testing loss')
plt.xlabel('Epochs')
plt.ylabel('Ethrophy Loss')
plt.legend(loc='upper right')
plt.show()


w_cal = sess.run(w)
b_cal = sess.run(b)

x1 = np.linspace(2,4.5,2)
x2 = (-(w_cal[0]*x1 + b_cal)/w_cal[1]).reshape(-1)

train_1 =(y_train==1).reshape(-1)
train_0 =(y_train==0).reshape(-1)
test_1 =(y_test==1).reshape(-1)
test_0 =(y_test==0).reshape(-1)


plt.plot(x_train[train_1,0],x_train[train_1,1],'*r',label='train_set, #1')
plt.plot(x_train[train_0,0],x_train[train_0,1],'*b',label='train_set, #0')
plt.plot(x_test[test_1,0],x_test[test_1,1],'or',label='test_set, #1')
plt.plot(x_test[test_0,0],x_test[test_0,1],'ob',label='test_set, #0')
plt.plot(x1,x2,'g-')
plt.xlabel('Feature #1')
plt.ylabel('Feature #2')
plt.legend(loc='upper right')
plt.show()