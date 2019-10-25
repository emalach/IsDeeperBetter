import argparse
import time
import datetime

import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from cantor import get_cantor_set
from ifs import get_sierpinsky_set, get_pentaflake_set, get_vicsek_set
import matplotlib.pyplot as plt


FRACTAL_DATASETS = ['cantor','sierpinsky', 'pentaflake','vicsek']

def weight_variable(shape, name, constant=False):
    initial = tf.truncated_normal(shape, stddev=1/np.sqrt(shape[0]))
    return tf.Variable(initial, name=name, trainable=not constant)

def bias_variable(shape, name, constant=False):
    initial = tf.constant(0.1, shape=shape) 
    return tf.Variable(initial, name=name, trainable=not constant)

class Layer:
    def __init__(self, input_, origin_, d, d_, k, index=0, activation='RELU'):
        self.input_ = input_
        self.origin_ = origin_
        d_o = self.origin_.shape.as_list()[1]

        self.d = d
        self.k = k

        self.W1 = weight_variable([d, k], '%d.W1' % index)
        self.b1 = bias_variable([k], '%d.B1' % index)
        self.h1 = tf.matmul(self.input_, self.W1) + self.b1
        self.activation = activation

        if activation == 'RELU' or activation == 'RESNET':
            self.h1 = tf.nn.relu(self.h1)

        self.W2 = weight_variable([k, d_], '%d.W2' % index)
        self.b2 = bias_variable([d_], '%d.B2' % index)
        self.h2 = tf.matmul(self.h1, self.W2) + self.b2

    def output(self):
        return self.h2

    def hidden(self):
        return self.h1

    def variables(self):
        if self.activation != 'GALUS':
            return [self.W1, self.b1, self.W2, self.b2]
        else:
            return [self.W1, self.b1, self.W2, self.b2, self.sigma]
        #return [self.W1, self.W2]
        #return [self.W1, self.b1]

    def get_train_step(self, labels, lr, optimization='e2e', var_list=None):
        #cross_entropy = tf.losses.softmax_cross_entropy(labels, self.output())
        # import pdb
        # pdb.set_trace()
        #loss = tf.reduce_mean((labels-self.output())**2)
        #loss = tf.reduce_mean((labels-self.output())**2)
        # loss = tf.reduce_mean(tf.log(1+tf.exp(-labels*self.output())))
        loss = tf.reduce_mean(tf.nn.relu(1-labels*self.output()))
        correct_prediction = tf.equal(tf.sign(self.output()), labels)
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        if optimization == 'e2e':
            if var_list is None:
                train_step = tf.train.AdamOptimizer(lr).minimize(loss)
            else:
                train_step = tf.train.AdamOptimizer(lr).minimize(loss, var_list=var_list)
        elif optimization == 'lbl':
            train_step = tf.train.AdamOptimizer(lr).minimize(loss, var_list=self.variables())
        return loss, accuracy, train_step


class Dataset:
    def __init__(self, dataset='fmnist', m=None, d=28**2, batch_size=100, pdeg_train=0, pdeg_val=0):
        self.batch_size = batch_size
        self.dataset = dataset
        self.m = m
        if self.dataset.startswith('cantor'):
            depth = int(self.dataset.replace('cantor', ''))
            self.dataset = 'cantor'
            self.data = {'train': None, 'test': None}
            for tt in self.data.keys():
                if tt == 'train':
                    pdeg = pdeg_train
                    m_ = m
                else:
                    pdeg = pdeg_val
                    m_ = m/10
                if pdeg < 10: # polynomial with degree pdeg
                    prob_vec = np.linspace(depth,1,depth)**pdeg
                else: #uniform on first pdeg-10 entries
                    th = pdeg-10
                    prob_vec = np.zeros(depth)
                    prob_vec[:th+1] = 1.0
                print('prob_vec=%s' % prob_vec)
                prob_vec /= np.sum(prob_vec)
                prob_vec = 1-0.5**prob_vec
                assert np.abs(np.prod(1-prob_vec)-0.5) < 1e-6
                X,Y = get_cantor_set(m_, d, depth, prob_vec)
                self.data[tt] = {'X': X, 'Y': Y}
        elif self.dataset.startswith('sierpinsky'):
            depth = int(self.dataset.replace('sierpinsky', ''))
            self.dataset = 'sierpinsky'
            self.data = {'train': None, 'test': None}
            for tt in self.data.keys():
                if tt == 'train':
                    pdeg = pdeg_train
                    m_ = m
                else:
                    pdeg = pdeg_val
                    m_ = m/10
                if pdeg < 10: # polynomial with degree pdeg
                    prob_vec = np.linspace(depth,1,depth)**pdeg
                else: #uniform on first pdeg-10 entries
                    th = pdeg-10
                    prob_vec = np.zeros(depth)
                    prob_vec[:th+1] = 1.0
                print('prob_vec=%s' % prob_vec)
                prob_vec /= np.sum(prob_vec)
                prob_vec = 1-0.5**prob_vec
                assert np.abs(np.prod(1-prob_vec)-0.5) < 1e-6
                X,Y = get_sierpinsky_set(m_, d, depth, prob_vec)
                self.data[tt] = {'X': X, 'Y': Y}
        elif self.dataset.startswith('pentaflake'):
            depth = int(self.dataset.replace('pentaflake', ''))
            self.dataset = 'pentaflake'
            self.data = {'train': None, 'test': None}
            for tt in self.data.keys():
                if tt == 'train':
                    pdeg = pdeg_train
                    m_ = m
                else:
                    pdeg = pdeg_val
                    m_ = m/10
                if pdeg < 10: # polynomial with degree pdeg
                    prob_vec = np.linspace(depth,1,depth)**pdeg
                else: #uniform on first pdeg-10 entries
                    th = pdeg-10
                    prob_vec = np.zeros(depth)
                    prob_vec[:th+1] = 1.0
                print('prob_vec=%s' % prob_vec)
                prob_vec /= np.sum(prob_vec)
                prob_vec = 1-0.5**prob_vec
                assert np.abs(np.prod(1-prob_vec)-0.5) < 1e-6
                X,Y = get_pentaflake_set(m_, d, depth, prob_vec)
                self.data[tt] = {'X': X, 'Y': Y}
        elif self.dataset.startswith('vicsek'):
            depth = int(self.dataset.replace('vicsek', ''))
            self.dataset = 'vicsek'
            self.data = {'train': None, 'test': None}
            for tt in self.data.keys():
                if tt == 'train':
                    pdeg = pdeg_train
                    m_ = m
                else:
                    pdeg = pdeg_val
                    m_ = m/10
                if pdeg < 10: # polynomial with degree pdeg
                    prob_vec = np.linspace(depth,1,depth)**pdeg
                else: #uniform on first pdeg-10 entries
                    th = pdeg-10
                    prob_vec = np.zeros(depth)
                    prob_vec[:th+1] = 1.0
                print('prob_vec=%s' % prob_vec)
                prob_vec /= np.sum(prob_vec)
                prob_vec = 1-0.5**prob_vec
                assert np.abs(np.prod(1-prob_vec)-0.5) < 1e-6
                X,Y = get_vicsek_set(m_, d, depth, prob_vec)
                self.data[tt] = {'X': X, 'Y': Y}


        self.step = 0

    def train_batch(self):
        if self.dataset in FRACTAL_DATASETS:
            if self.step == 0:
                self.idx = np.random.permutation(self.m)
            X = self.data['train']['X'][self.idx[self.step*self.batch_size:(self.step+1)*self.batch_size]]
            Y = self.data['train']['Y'][self.idx[self.step*self.batch_size:(self.step+1)*self.batch_size]]

            self.step = (self.step + 1) % (self.m/self.batch_size)
        return X, Y

    def test_batch(self):
        if self.dataset in FRACTAL_DATASETS:
            # train=test
            X = self.data['test']['X']
            Y = self.data['test']['Y']
        return X, Y

def get_net(X_, d, d_, k, T, activation='RELU'):
    layers = []
    for i in range(T):
        if i == 0:
            cur_layer = Layer(X_, X_, d, d_, k, i, activation=activation)
        elif i < T-1 or activation != 'RESNET':
            cur_layer = Layer(layers[-1].hidden(), X_, k, d_, k, i, activation=activation)
        elif activation == 'RESNET':
            sum_layers = tf.reduce_sum([l.hidden() for l in layers], axis=0)
            cur_layer = Layer(sum_layers, X_, k, d_, k, i, activation=activation)

        layers.append(cur_layer)

    return layers

def train_net(m, k, T, d, lr, pdeg_train=0, pdeg_val=0, optimization='e2e', dataset_name='fmnist', activation='RELU'):
    config = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1,
                           allow_soft_placement=True, device_count = {'CPU': 1})
    d_ = 1
    batch_size = 100
    dataset = Dataset(dataset_name, m, d, batch_size, pdeg_train=pdeg_train, pdeg_val=pdeg_val)
    x_ = tf.placeholder(tf.float32, [None, d]) 
    y_ = tf.placeholder(tf.float32, [None, d_])

    num_params = d*k + k + (T-1)*(k**2+k) + k + 1
    print '#params: %d' % num_params

    net = get_net(x_, d, d_, k, T, activation=activation)  

    all_net_vars = sum([net[i].variables() for i in range(T)], [])
    e2e_loss, e2e_accuracy, e2e_train_step = net[-1].get_train_step(y_, lr, optimization='e2e', var_list=all_net_vars)
    lbl_loss = []
    lbl_accuracy = []
    lbl_train_step = []
    for layer in net:
        cur_loss, cur_accuracy, cur_train_step = layer.get_train_step(y_, lr, optimization='lbl')
        lbl_loss.append(cur_loss)
        lbl_accuracy.append(cur_accuracy)
        lbl_train_step.append(cur_train_step)

    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())

        #max_train_steps = [100000, 200000]
        max_train_steps = [10**6]*T
        print_train_step = 10000
        layers_to_iter = 1 if optimization == 'e2e' else T
        for layer in range(layers_to_iter):
            total_steps = 0
            print 'layer', layer
            if optimization == 'e2e':
                train_step = e2e_train_step
                loss = e2e_loss
                accuracy = e2e_accuracy
            elif optimization == 'lbl':
                train_step = lbl_train_step[layer]
                loss = lbl_loss[layer]
                accuracy = lbl_accuracy[layer]
            loss_train = 0
            acc_train = 0
            for total_steps in range(max_train_steps[layer]):
                batch_xs, batch_ys = dataset.train_batch()
                _, loss_train_, acc_train_ = sess.run([train_step, loss, accuracy], feed_dict={x_: batch_xs, y_: batch_ys})
                loss_train += loss_train_
                acc_train += acc_train_
                total_steps += 1
                if total_steps % print_train_step == 0:
                    loss_train /= print_train_step
                    acc_train /= print_train_step
                    X, Y = dataset.test_batch()
                    loss_val, acc_val = sess.run([loss, accuracy], feed_dict={x_: X, y_: Y})
                    ts = time.time()
                    st = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')
                    print st, total_steps, loss_val, acc_val, loss_train, acc_train
                    loss_train = 0
                    acc_train = 0
                if total_steps > max_train_steps:
                    break


parser = argparse.ArgumentParser(description='Run ReLU Networks.')
parser.add_argument('--k', type=int, default=20,
                    help='#neurons')
parser.add_argument('--T', type=int, default=5,
                    help='net depth')
parser.add_argument('--d', type=int, default=5,
                    help='dimension')
parser.add_argument('--m', type=int, default=50000,
                    help='dimension')
parser.add_argument('--pdeg_train', type=int, default=0,
                    help='profile degree train')
parser.add_argument('--pdeg_val', type=int, default=0,
                    help='profile degree val')
parser.add_argument('--lr', type=float, default=0.005,
                    help='learning rate')
parser.add_argument('--activation', type=str, default='RELU',
                    help='activation (RELU)')
parser.add_argument('--opt', type=str, default='e2e',
                    help='optimization (e2e/lbl)')
parser.add_argument('--dataset', type=str, default='cantor5',
                    help='dataset')



args = parser.parse_args()
print 'Args:', args

k = args.k
d = args.d
T = args.T
lr = args.lr
m = args.m
optimization = args.opt
activation = args.activation
dataset = args.dataset

train_net(m, k, T, d, lr, pdeg_train=args.pdeg_train, pdeg_val=args.pdeg_val, optimization=optimization, dataset_name=dataset, activation=activation)
