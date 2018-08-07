import numpy as np
import tensorflow as tf
from PIL import Image

def add_substr(a, sign, b):
    if sign > 0:
        return a + b
    else:
        return a - b

def get_sign(s):
    if s > 0:
        return '+'
    else:
        return '-'


def onezero_bin(a, b):
    out = []
    if a == 0:
        out.append(0)
        out.append(0)
    else:
        out.append(1)
        out.append(1)

    if b == 0:
        out.append(0)
        out.append(0)
    else:
        out.append(1)
        out.append(1)


    return out

def data(n=1000):
    tsts = np.zeros(shape=(2*2*2*50*6*10, 4, 4*8))
    y = np.zeros(shape=(2*2*2*50*6*10, 8))
    opp_i_s = []
    ops_map = {}
    ops = np.zeros(shape=(2*2*2*50*6*10))
    y_int = np.zeros(shape=(2*2*2*50*6*10))
    e = 0
    for sign1 in range(2):
        op = sign1
        for sign2 in range(2):
            op += 1
            for sign3 in range(2):
                op += 1
                if sign1 > 0:
                    rng = range(36, 86)
                else:
                    rng = range(46, 96)
                for bias in range(1, 7):
                    op += 1
                    opp_i = op - 3
                    opp_i_s.append(opp_i)
                    ops_map[opp_i] = ' '.join(['a', get_sign(sign2), 'b', get_sign(sign3), str(bias)])
                    for a_0 in rng:

                        a_i = {}
                        a = np.zeros(shape=(4, 8))
                        b = np.zeros(shape=(4, 8))
                        c = np.zeros(shape=(4, 8))
                        opp = np.zeros(shape=(4, 8))
                        a_i[-1] = a_0
                        for b_i in range(3, 13):
                            c_i = -777
                            for i in range(0, 4):
                                a_i[i] = add_substr(a_i[i - 1], sign1, np.random.randint(low=1, high=3))
                                c_i = add_substr(add_substr(a_i[i], sign2, b_i), sign3, bias)
                                a[i] = np.array(list("{0:b}".format(a_i[i]).zfill(8)), dtype=int)
                                b[i] = np.array(list("{0:b}".format(b_i).zfill(8)), dtype=int)
                                c[i] = np.array(list("{0:b}".format(c_i).zfill(8)), dtype=int)
                                opp[i] = np.array(list("{0:b}".format(opp_i).zfill(8)), dtype=int)

                            ops[e] = op-3

                            y_int[e] = c_i

                            y[e] = c[-1]
                            asas = c
                            asas[-1] = np.zeros(8)

                            tsts[e] = np.array([np.reshape(a, (4*8)),
                                                np.reshape(opp, (4*8)),
                                                np.reshape(b, (4*8)),
                                                np.reshape(asas, (4*8))])


                            e += 1

    print(len(tsts))
    print(list(set(opp_i_s)))

    return tsts, y, ops_map, ops, y_int


x, y, lbls_map, lbls, y_ints = data()

print(x.shape)

indx = np.arange(x.shape[0])
np.random.shuffle(indx)
x = x[indx]
y = y[indx]
lbls = lbls[indx]
y_ints = y_ints[indx]

val_x = x[:BATCH_SIZE]
train_x = x[BATCH_SIZE:]

val_y = y[:BATCH_SIZE]
train_y = y[BATCH_SIZE:]

val_lbls = lbls[:BATCH_SIZE]
train_lbls = lbls[BATCH_SIZE:]

val_y_ints = y_ints[:BATCH_SIZE]
train_y_ints = y_ints[BATCH_SIZE:]


EPOCHS = 500*2
IMG_SZ = 4*8
N_FRAMES = 4
N_BATCHES = 1000
BATCH_SIZE = 128


# Define some tf.placeholders
x = tf.placeholder(tf.float32, shape=[None, N_FRAMES, IMG_SZ], name='Input')
y = tf.placeholder(tf.float32, shape=[None, 8], name='Output')
images = tf.reshape(x, [BATCH_SIZE*N_FRAMES, IMG_SZ], name='images')

fc1w = tf.Variable(tf.truncated_normal([IMG_SZ, 64], dtype=tf.float32, stddev=1e-1), name='weights')
fc1b = tf.Variable(tf.constant(0.0, shape=[64], dtype=tf.float32), trainable=True, name='biases')
fc1l = tf.nn.bias_add(tf.matmul(images, fc1w), fc1b)
fc1l = tf.contrib.layers.batch_norm(fc1l, center=True, scale=True)
fc1 = tf.nn.relu6(fc1l)

lstm_in = tf.reshape(fc1, [BATCH_SIZE, N_FRAMES, 64])
rnn = tf.contrib.rnn.GRUCell(64) #tf.contrib.rnn
rnns = tf.contrib.rnn.MultiRNNCell([rnn for _ in range(1)])
out, _ = tf.nn.dynamic_rnn(rnns, lstm_in, dtype=tf.float32)

lstm_out = tf.reshape(out, [BATCH_SIZE*N_FRAMES, 64])
lstm_out = tf.nn.relu6(lstm_out)

fc2w = tf.Variable(tf.truncated_normal([64, 8], dtype=tf.float32, stddev=1e-1), name='weights')
fc2b = tf.Variable(tf.constant(0.0, shape=[8], dtype=tf.float32), trainable=True, name='biases')
fc2l = tf.nn.bias_add(tf.matmul(lstm_out, fc2w), fc2b)
fc2 = tf.nn.sigmoid(fc2l)
output_ = tf.reshape(fc2, [BATCH_SIZE, N_FRAMES, 8])
output = output_[:, -1, :]

# Define some tf operations
loss = tf.reduce_mean(tf.losses.sigmoid_cross_entropy(logits=output, multi_class_labels=y))
optimizer = tf.train.AdamOptimizer(learning_rate=0.0001).minimize(loss)


print('Training...')
with tf.Session() as sess:

    sess.run(tf.global_variables_initializer())

    for i in range(EPOCHS):
        print('Epoch:', i)

        for j in range(N_BATCHES):
            # print(train_x.shape[0], BATCH_SIZE)
            randidx = np.random.randint(train_x.shape[0], size=BATCH_SIZE)
            batch_xs = train_x[randidx]
            batch_ys = train_y[randidx]
            opt_res = sess.run([optimizer], feed_dict={x: batch_xs, y: batch_ys})

            if (j+1) % 1000 == 0:
                idx = np.random.randint(train_x.shape[0], size=BATCH_SIZE)
                batch_xs = train_x[idx]
                batch_ys = train_y[idx]
                l = sess.run(loss, feed_dict={x: batch_xs, y: batch_ys})

                idx = np.random.randint(val_x.shape[0], size=BATCH_SIZE)
                batch_xs = val_x[idx]
                batch_ys = val_y[idx]
                vl = sess.run(loss, feed_dict={x: batch_xs, y: batch_ys})
                print(str(j + 1) + '/' + str(N_BATCHES) + ' ', '\tt_loss =', str(l)[:6], '\tv_loss =',  str(vl)[:6])


        idx = np.random.randint(val_x.shape[0], size=BATCH_SIZE)
        b_xs = val_x[idx]
        b_ys = val_y[idx]
        b_ls = val_lbls[idx]
        b_yis = val_y_ints[idx]
        a = sess.run(output, feed_dict={x: b_xs, y: b_ys})
        xx = b_xs[0]
        yy = b_ys[0]
        ll = b_ls#[0]
        yi = b_yis#[0]

        # print([str(int(a)) for a in list(np.round(a[0]))], [a for a in list(np.round(a[0]))], list(np.round(a[0])), np.round(a[0]))
        # print(''.join([str(a) for a in list(np.round(a[0]))]).lstrip('0'))
        g = []
        h = []
        for k in range(a.shape[0]):
            # print('y:', int(yi[k]))
            if len(str(int(yi[k]))) == 0:
                assert 1 == 0
            if int(''.join([str(int(d)) for d in list(np.round(a[k]))]), 2) != 0:
                A = int(''.join([str(int(d)) for d in list(np.round(a[k]))]).lstrip('0'), 2)
            else:
                A = 0

            g.append(A)
            h.append(int(yi[k]))
            # print(lbls_map[ll[k]], int(yi[k]), A)

        q = [int(q_) for q_ in [g_ == h_ for (g_, h_) in zip(g, h)]]
        print('Accuracy:', np.sum(q)/val_x.shape[0])
        
