import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
from sklearn import decomposition
from time_series_window import *
from tqdm import tqdm


data = '/media/mcamp/Local SSHD/PythonProjects/Datasets/TwoSigma/train.h5'
#data = '/media/mcamp/HDD/PythonProjects/Datasets/train.h5'
store = pd.HDFStore(data)
df = store['train']
vars_to_use = ['fundamental_41',
 'fundamental_17',
 'technical_18',
 'fundamental_58',
 'fundamental_19',
 'timestamp',
 'technical_34',
 'fundamental_33',
 'technical_11',
 'fundamental_27',
 'technical_3',
 'fundamental_59',
 'technical_29',
 'fundamental_10',
 'technical_19',
 'technical_25',
 'technical_38',
 'fundamental_40',
 'technical_33',
 'technical_16',
 'technical_41',
 'technical_7',
 'technical_43',
 'technical_2',
 'fundamental_42',
 # 'fundamental_0', #only removed these to make a square input shape
 'technical_20',
 'technical_37',
 'fundamental_32',
 'technical_17',
 'technical_28',
 'technical_32',
 #'technical_13',
 'technical_31',
 'fundamental_25',
 'technical_0',
 'id',
 'technical_14',
 'technical_12',
 'fundamental_62',
 'technical_39',
 'technical_36',
 'fundamental_21',
 'fundamental_48',
 'fundamental_53',
 'technical_9',
 'technical_5',
 'technical_40',
 'technical_35',
 'derived_1',
 'derived_3',
 'technical_24',
 'fundamental_52',
 'technical_44',
 'technical_1',
 'technical_30',
 'technical_21',
 'fundamental_12',
 'derived_0',
 'fundamental_18',
 'technical_42',
 'technical_27',
 'technical_10',
 'technical_6',
 'technical_22',
 'y']

df = df[vars_to_use]
df = df.sort_values('timestamp', 0)
df = df.apply(lambda x: x.fillna(x.mean()),axis=0)

window_size = 65

train = df.iloc[:-100000,:].reset_index(drop=True)
test = df.iloc[-100000:,:].reset_index(drop=True)
win_train = time_series_windows(train, window_size)
win_test = time_series_windows(test, window_size)
features = (window_size*train.shape[1])-train.shape[1]

n_classes = 1
batch_size = 256

x = tf.placeholder('float', [None, features])
y = tf.placeholder('float')

keep_rate = 0.5
keep_prob = tf.placeholder(tf.float32)


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def maxpool2d(x):
    #                        size of window         movement of window
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


def convolutional_neural_network(x):
    weights = {'W_conv1': tf.Variable(tf.random_normal([5, 5, 1, 32])),
               'W_conv2': tf.Variable(tf.random_normal([5, 5, 32, 64])),
               'W_fc': tf.Variable(tf.random_normal([16*16*64, 40])),
               'W_fc2': tf.Variable(tf.random_normal([40, 40])),
               'out': tf.Variable(tf.random_normal([40, n_classes]))}

    biases = {'b_conv1': tf.Variable(tf.random_normal([32])),
              'b_conv2': tf.Variable(tf.random_normal([64])),
              'b_fc': tf.Variable(tf.random_normal([40])),
              'b_fc2': tf.Variable(tf.random_normal([40])),
              'out': tf.Variable(tf.random_normal([n_classes]))}

    x0 = tf.reshape(x, shape=[-1, 64, 64, 1])
    with tf.name_scope('Conv_Layer1'):
        conv1 = tf.nn.relu(conv2d(x0, weights['W_conv1']) + biases['b_conv1'])
        conv1 = maxpool2d(conv1)
    with tf.name_scope('Conv_Layer2'):
        conv2 = tf.nn.relu(conv2d(conv1, weights['W_conv2']) + biases['b_conv2'])
        conv2 = maxpool2d(conv2)
    with tf.name_scope('FC_Layer1'):
        fc = tf.reshape(conv2, [-1, 16 * 16 * 64])
        fc = tf.nn.relu(tf.matmul(fc, weights['W_fc']) + biases['b_fc'])
        fc = tf.nn.dropout(fc, keep_rate)
    with tf.name_scope('FC_Layer2'):
        fc2 = tf.nn.relu(tf.matmul(fc, weights['W_fc2']) + biases['b_fc2'])
        fc2 = tf.nn.dropout(fc2, keep_rate)

        output = tf.matmul(fc2, weights['out']) + biases['out']

    return output


def train_neural_network(x):
    prediction = convolutional_neural_network(x)
    mean_percent_error = tf.reduce_mean(tf.abs(tf.div(tf.sub(y, prediction), y)))
    cost = tf.sqrt(tf.reduce_mean(tf.square(tf.sub(y, prediction))))
    tf.summary.scalar('RMSE', cost)
    optimizer = tf.train.AdamOptimizer(learning_rate=0.0051).minimize(mean_percent_error)
    tf.summary.scalar('MAPE', mean_percent_error)
    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    hm_epochs = 10000
    with tf.Session() as sess:
        sess.run(init_op)
        train_writer = tf.summary.FileWriter('./train', sess.graph)
        test_writer = tf.summary.FileWriter('./test', sess.graph)
        saver = tf.train.Saver(max_to_keep=3)
        ckpt = tf.train.get_checkpoint_state("./")
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
        merged = tf.summary.merge_all()

        for epoch in range(hm_epochs):
            print('Epoch', epoch)
            epoch_loss = 0
            mini_batch = 0
            for _ in tqdm(range(int(train.shape[0] / batch_size))):
                mini_batch += 1
                train_batch = win_train.make_batch(batch_size)
                epoch_x, epoch_y = train_batch[:,: -train.shape[1]], train_batch[:, -1]
                summary, _, c, mpe = sess.run([merged, optimizer, cost, mean_percent_error],
                                         feed_dict={x: epoch_x, y: epoch_y, keep_prob: 0.5})
                epoch_loss += c
                train_writer.add_summary(summary, epoch)
                if mini_batch % 100 == 0:
                    print('RMSE:', c)
                    print('MAPE:', mpe)
                    print('Saving model...')
                    save_path = saver.save(sess, './model.ckpt', epoch)
                if mini_batch % 2000 == 0:
                    test_batch = win_test.make_batch(batch_size)
                    summary, yhat = sess.run([merged, prediction], feed_dict={
                        x: test_batch[:, :-train.shape[1]], y: test_batch[:, -1], keep_prob: 1.0})
                    test_writer.add_summary(summary, epoch)
                    correct = tf.sqrt(tf.reduce_mean(tf.square(tf.sub(y, prediction))))
                    accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
                    print('MSE:', accuracy.eval({x: test_batch[:, :-train.shape[1]], y: test_batch[:, -1]}))
                    print('yhat:', yhat, 'y:', test_batch[:, -1])
            print('Epoch', epoch, 'completed out of', hm_epochs, 'loss:', epoch_loss)
            if epoch % 10 == 0:
                test_batch = win_test.make_batch(batch_size)
                summary, yhat = sess.run([merged, prediction], feed_dict={
                    x: test_batch[:,:-train.shape[1]], y: test_batch[:,-1], keep_prob: 1.0})
                test_writer.add_summary(summary, epoch)
                correct = tf.sqrt(tf.reduce_mean(tf.square(tf.sub(y, prediction))))
                accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
                print('MSE:', accuracy.eval({x: test_batch[:,:-train.shape[1]], y: test_batch[:,-1]}))
            if epoch % 5 == 0:
                print('Saving model...')
                save_path = saver.save(sess, './model.ckpt', epoch)
            print('RMSE:', c)
        test_batch = win_test.make_batch(batch_size)
        correct = tf.sqrt(tf.reduce_mean(tf.square(tf.sub(y, prediction))))
        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
        print('MSE:', accuracy.eval({x: test_batch[:,:-train.shape[1]], y: test_batch[:,-1]}))


train_neural_network(x)