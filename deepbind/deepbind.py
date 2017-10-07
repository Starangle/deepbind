import tensorflow as tf
from tensorflow.contrib import rnn
import numpy
import time


class Data:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def next_batch(self, batch_size):
        idx = numpy.arange(0, len(self.x))
        numpy.random.shuffle(idx)
        idx = idx[0:batch_size]
        rx = [self.x[i] for i in idx]
        ry = [self.y[i] for i in idx]
        return numpy.array(rx), numpy.array(ry)

    def all(self):
        return numpy.asarray(self.x), numpy.asarray(self.y)


class LoadData:
    @staticmethod
    def load(source_file):
        x = list()
        y = list()
        with open(source_file) as f:
            for line in f.readlines():
                seq = list()
                words = line.split()

                for ch in words[1]:
                    x_vector = [0, 0, 0, 0]
                    if ch == 'A':
                        x_vector[0] = 1
                    elif ch == 'T':
                        x_vector[1] = 1
                    elif ch == 'G':
                        x_vector[2] = 1
                    else:
                        x_vector[3] = 1
                    seq.append(x_vector)
                    # seq = numpy.reshape(seq[0:100], [20, 20])
                x.append(seq)

                y_vector = [0, 0]
                if words[2] == '0':
                    y_vector[0] = 1
                else:
                    y_vector[1] = 1
                y.append(y_vector)
        return Data(numpy.asarray(x), numpy.asarray(y))


class Manager:
    def __init__(self, task_list_file):
        self.tasks = open(task_list_file).readlines()
        self.index = 0

    def next_task(self):
        if self.index < len(self.tasks):
            folder = self.tasks[self.index][0:-1]
            self.index += 1
            return [folder + "train.data", folder + "test.data"]
        else:
            return None


def run_rnn():
    batch_size = 512
    learning_rate = 0.002
    input_size = 4
    time_step = 101
    output_size = 2
    hidden_size = 128
    epoch = 5
    all_data_epoch = 30
    deepth = 6
    result_name = "result.txt"

    xp = tf.placeholder(tf.float32, [None, time_step, input_size])
    yp = tf.placeholder(tf.float32, [None, output_size])
    weight = tf.Variable(tf.random_normal([hidden_size, output_size]), name="weight")
    bias = tf.Variable(tf.random_normal([output_size]), name="bias")

    train_xpu = tf.unstack(xp, time_step, 1)
    cells = list()
    for i in range(deepth):
        lstm_cell = rnn.LSTMCell(hidden_size, forget_bias=1.0,activation=tf.nn.softsign)
        # lstm_cell_dropout = rnn.DropoutWrapper(lstm_cell, 0.8, 0.8)
        cells.append(lstm_cell)
    lstm = rnn.MultiRNNCell(cells)

    outputs, states = rnn.static_rnn(lstm, train_xpu, dtype=tf.float32)
    predict = tf.matmul(outputs[-1], weight) + bias
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=predict, labels=yp))
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

    correct = tf.equal(tf.argmax(predict, 1), tf.argmax(yp, 1))
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
    auc = tf.metrics.auc(tf.argmax(predict, 1), tf.argmax(yp, 1), )

    manager = Manager("data_list.txt")
    open(result_name, "w+").close()

    task = manager.next_task()
    while task is not None:
        out_name = task[0].split("\\")[1]
        train_data_set = LoadData.load(task[0])
        test_data_set = LoadData.load(task[1])
        start_time = time.time()

        if all_data_epoch is not None:
            epoch = int(all_data_epoch * train_data_set.x.shape[0] / batch_size)

        # fix the value of batch_size to avoid out of range
        batch_size = min(batch_size, train_data_set.x.shape[0])

        init = tf.global_variables_initializer()
        with tf.Session() as sess:
            sess.run(init)
            for i in range(epoch):
                input_x, input_y = train_data_set.next_batch(batch_size)
                sess.run(optimizer, feed_dict={xp: input_x, yp: input_y})
                acc = sess.run(accuracy, feed_dict={xp: input_x, yp: input_y})
                loss = sess.run(cost, feed_dict={xp: input_x, yp: input_y})
                print("Training " + out_name + ", Epoch " + str(i + 1) + " of " + str(
                    epoch) + ", Minibatch Loss= " + "{:.6f}".format(
                    loss) + ", Training Accuracy= " + "{:.5f}".format(acc))

            input_x, input_y = test_data_set.all()
            sess.run(tf.local_variables_initializer())
            test_acc = sess.run(accuracy, feed_dict={xp: input_x, yp: input_y})
            test_auc = sess.run(auc, feed_dict={xp: input_x, yp: input_y})[1]

            print("Testing Accuracy:", test_acc)
            print("Testing Auc:", test_auc)

            # record
            with open(result_name, "a") as f:
                f.write(out_name + " " + str(test_acc) + " " + str(test_auc) + '\n')

            end_time = time.time()
            print("Time used {:.2f} minutes".format((end_time - start_time) / 60))

            print("Saving...")
            saver = tf.train.Saver()
            saver.save(sess, "model\\" + out_name + ".ckpt")
            print("Saved!")

        task = manager.next_task()


if __name__ == '__main__':
    run_rnn()
