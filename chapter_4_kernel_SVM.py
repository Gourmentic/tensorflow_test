import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import StratifiedShuffleSplit

digits_data = datasets.load_digits()
feature_data = digits_data['data']
label_data = digits_data['target']
label_table = np.zeros(shape=(len(label_data),10))

for i in range(len(label_data)):

    label_table[i,label_data[i]] = 1

# plt.imshow(digits_data['images'][-1],cmap=plt.cm.gray_r,interpolation='nearest')
# plt.show()

# split the train and test dataset
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.3)

for train_index, test_index in sss.split(feature_data, label_data):

    train_x, train_y = feature_data[train_index], label_table[train_index]
    test_x, test_y = feature_data[test_index], label_table[test_index]


X_input = tf.placeholder(shape=(None,64),dtype=tf.float32)
Y_target = tf.placeholder(shape=(None,10))


#


#the instance should be created throught dual problem
#one vs others will be used for mutli-classification



