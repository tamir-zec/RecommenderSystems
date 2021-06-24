'''
NARRE
@author:
Chong Chen (cstchenc@163.com)

@ created:
27/8/2017
@references:

'''
import datetime
import os
import pickle
import sys
from gensim.models import KeyedVectors

import numpy as np
import tensorflow as tf

import NARRE

# Model Hyperparameters
tf.compat.v1.flags.DEFINE_integer("embedding_dim", 300, "Dimensionality of character embedding ")
tf.compat.v1.flags.DEFINE_string("filter_sizes", "3", "Comma-separated filter sizes")
tf.compat.v1.flags.DEFINE_integer("num_filters", 100, "Number of filters per filter size")
tf.compat.v1.flags.DEFINE_float("dropout_rate", 0.5, "Dropout keep probability")
tf.compat.v1.flags.DEFINE_float("l2_reg_lambda", 0.001, "L2 regularization lambda")
# Training parameters
tf.compat.v1.flags.DEFINE_integer("batch_size", 100, "Batch Size ")
tf.compat.v1.flags.DEFINE_integer("num_epochs", 40, "Number of training epochs ")
# Misc Parameters
tf.compat.v1.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.compat.v1.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")


def train_step(u_batch, i_batch, uid, iid, reuid, reiid, y_batch, batch_num):
    """
    A single training step
    """
    feed_dict = {
        deep.input_u: u_batch,
        deep.input_i: i_batch,
        deep.input_uid: uid,
        deep.input_iid: iid,
        deep.input_y: y_batch,
        deep.input_reuid: reuid,
        deep.input_reiid: reiid,
        deep.drop0: 0.8,
        deep.dropout_rate: FLAGS.dropout_rate
    }
    _, step, loss, accuracy, mae, u_a, i_a, fm = sess.run(
        [train_op, global_step, deep.loss, deep.accuracy, deep.mae, deep.u_a, deep.i_a, deep.score],
        feed_dict)
    time_str = datetime.datetime.now().isoformat()
    # print("{}: step {}, loss {:g}, rmse {:g},mae {:g}".format(time_str, batch_num, loss, accuracy, mae))
    return accuracy, mae, u_a, i_a, fm


def dev_step(u_batch, i_batch, uid, iid, reuid, reiid, y_batch, writer=None):
    """
    Evaluates model on a dev set

    """
    feed_dict = {
        deep.input_u: u_batch,
        deep.input_i: i_batch,
        deep.input_y: y_batch,
        deep.input_uid: uid,
        deep.input_iid: iid,
        deep.input_reuid: reuid,
        deep.input_reiid: reiid,
        deep.drop0: 1.0,
        deep.dropout_rate: 1.0
    }
    step, loss, accuracy, mae = sess.run(
        [global_step, deep.loss, deep.accuracy, deep.mae],
        feed_dict)
    time_str = datetime.datetime.now().isoformat()
    # print("{}: step{}, loss {:g}, rmse {:g},mae {:g}".format(time_str, step, loss, accuracy, mae))

    return [loss, accuracy, mae]


if __name__ == '__main__':

    categories = ['kindle', 'movies', 'toys']
    for category in categories:
        TPS_DIR = '../../data/' + category
        tf.compat.v1.flags.DEFINE_string("word2vec", os.path.join("../../data/", "google.bin"),
                                         "Word2vec file with pre-trained embeddings (default: None)")
        tf.compat.v1.flags.DEFINE_string("valid_data", os.path.join(TPS_DIR, category + '.test'),
                                         "Data for validation")
        tf.compat.v1.flags.DEFINE_string("para_data", os.path.join(TPS_DIR, category + '.para'), "Data parameters")
        tf.compat.v1.flags.DEFINE_string("train_data", os.path.join(TPS_DIR, category + '.train'), "Data for training")

        FLAGS = tf.compat.v1.flags.FLAGS
        FLAGS(sys.argv)
        # print("\nParameters:")
        # for attr, value in sorted(FLAGS.__flags.items()):
        #     print("{}={}".format(attr.upper(), value))
        # print("")

        print("Loading data...")
        pkl_file = open(FLAGS.para_data, 'rb')

        para = pickle.load(pkl_file)
        user_num = para['user_num']
        item_num = para['item_num']
        review_num_u = para['review_num_u']
        review_num_i = para['review_num_i']
        review_len_u = para['review_len_u']
        review_len_i = para['review_len_i']
        vocabulary_user = para['user_vocab']
        vocabulary_item = para['item_vocab']
        train_length = para['train_length']
        test_length = para['test_length']
        u_text = para['u_text']
        i_text = para['i_text']

        random_seed = 2017
        np.random.seed(random_seed)
        print(f'user_num: {user_num}')
        print(f'item_num: {item_num}')
        print(review_num_u)
        print(review_len_u)
        print(review_num_i)
        print(review_len_i)

        with tf.compat.v1.Graph().as_default():
            session_conf = tf.compat.v1.ConfigProto(
                allow_soft_placement=FLAGS.allow_soft_placement,
                log_device_placement=FLAGS.log_device_placement)
            session_conf.gpu_options.allow_growth = True
            sess = tf.compat.v1.Session(config=session_conf)
            with sess.as_default():
                deep = NARRE.NARRE(
                    review_num_u=review_num_u,
                    review_num_i=review_num_i,
                    review_len_u=review_len_u,
                    review_len_i=review_len_i,
                    user_num=user_num,
                    item_num=item_num,
                    num_classes=1,
                    user_vocab_size=len(vocabulary_user),
                    item_vocab_size=len(vocabulary_item),
                    embedding_size=FLAGS.embedding_dim,
                    embedding_id=32,
                    filter_sizes=list(map(int, FLAGS.filter_sizes.split(","))),
                    num_filters=FLAGS.num_filters,
                    l2_reg_lambda=FLAGS.l2_reg_lambda,
                    attention_size=32,
                    n_latent=32)
                tf.random.set_seed(random_seed)
                print(user_num)
                print(item_num)
                global_step = tf.Variable(0, name="global_step", trainable=False)

                # optimizer = tf.train.AdagradOptimizer(learning_rate=0.01, initial_accumulator_value=1e-8).minimize(deep.loss)
                optimizer = tf.compat.v1.train.AdamOptimizer(0.002, beta1=0.9, beta2=0.999, epsilon=1e-8).minimize(
                    deep.loss)

                train_op = optimizer  # .apply_gradients(grads_and_vars, global_step=global_step)

                sess.run(tf.compat.v1.global_variables_initializer())

                saver = tf.compat.v1.train.Saver()

                if FLAGS.word2vec:
                    # Load pretrained model (since intermediate data is not included, the model cannot be refined with additional data)
                    w2v = KeyedVectors.load_word2vec_format('../../data/GoogleNews-vectors-negative300.bin.gz', binary=True)
                    # Initial matrix with random uniform
                    initW = np.random.uniform(-1.0, 1.0, (len(vocabulary_user), FLAGS.embedding_dim))

                    for word in vocabulary_user:
                        if word in w2v:
                            idx = vocabulary_user[word]
                            initW[idx] = w2v[word]

                    sess.run(deep.W1.assign(initW))

                    initW = np.random.uniform(-1.0, 1.0, (len(vocabulary_item), FLAGS.embedding_dim))
                    for word in vocabulary_item:
                        if word in w2v:
                            idx = vocabulary_item[word]
                            initW[idx] = w2v[word]

                    sess.run(deep.W2.assign(initW))

                epoch = 1
                best_mae = 5
                best_rmse = 5
                train_mae = 0
                train_rmse = 0

                pkl_file = open(FLAGS.train_data, 'rb')

                train_data = pickle.load(pkl_file)

                train_data = np.array(train_data)
                pkl_file.close()

                pkl_file = open(FLAGS.valid_data, 'rb')

                test_data = pickle.load(pkl_file)
                test_data = np.array(test_data)
                pkl_file.close()

                data_size_train = len(train_data)
                data_size_test = len(test_data)
                batch_size = FLAGS.batch_size
                ll = int(len(train_data) / batch_size)
                for epoch in range(40):
                    # Shuffle the data at each epoch
                    shuffle_indices = np.random.permutation(np.arange(data_size_train))
                    shuffled_data = train_data[shuffle_indices]
                    for batch_num in range(ll):

                        start_index = batch_num * batch_size
                        end_index = min((batch_num + 1) * batch_size, data_size_train)
                        data_train = shuffled_data[start_index:end_index]

                        uid, iid, reuid, reiid, y_batch = zip(*data_train)
                        u_batch = []
                        i_batch = []
                        for i in range(len(uid)):
                            u_batch.append(u_text[uid[i][0]])
                            i_batch.append(i_text[iid[i][0]])
                        u_batch = np.array(u_batch)
                        i_batch = np.array(i_batch)

                        t_rmse, t_mae, u_a, i_a, fm = train_step(u_batch, i_batch, uid, iid, reuid, reiid, y_batch,
                                                                 batch_num)
                        current_step = tf.compat.v1.train.global_step(sess, global_step)
                        train_rmse += t_rmse
                        train_mae += t_mae
                        if batch_num % 500 == 0 and batch_num > 1:
                            print("\nEvaluation:")
                            print(batch_num)

                            loss_s = 0
                            accuracy_s = 0
                            mae_s = 0

                            ll_test = int(len(test_data) / batch_size) + 1
                            for batch_num in range(ll_test):
                                start_index = batch_num * batch_size
                                end_index = min((batch_num + 1) * batch_size, data_size_test)
                                data_test = test_data[start_index:end_index]

                                userid_valid, itemid_valid, reuid, reiid, y_valid = zip(*data_test)
                                u_valid = []
                                i_valid = []
                                for i in range(len(userid_valid)):
                                    u_valid.append(u_text[userid_valid[i][0]])
                                    i_valid.append(i_text[itemid_valid[i][0]])
                                u_valid = np.array(u_valid)
                                i_valid = np.array(i_valid)

                                loss, accuracy, mae = dev_step(u_valid, i_valid, userid_valid, itemid_valid, reuid,
                                                               reiid,
                                                               y_valid)
                                loss_s = loss_s + len(u_valid) * loss
                                accuracy_s = accuracy_s + len(u_valid) * np.square(accuracy)
                                mae_s = mae_s + len(u_valid) * mae
                            print(
                                f'loss_valid: {loss_s / test_length}, rmse_valid: {np.sqrt(accuracy_s / test_length)}, mae_valid: {mae_s / test_length}')
                            rmse = np.sqrt(accuracy_s / test_length)
                            mae = mae_s / test_length
                            if best_rmse > rmse:
                                best_rmse = rmse
                            if best_mae > mae:
                                best_mae = mae
                            print("")

                    print(str(epoch) + ':\n')
                    print("\nEvaluation:")
                    print("train:rmse,mae:", train_rmse / ll, train_mae / ll)
                    u_a = np.reshape(u_a[0], (1, -1))
                    i_a = np.reshape(i_a[0], (1, -1))

                    print(u_a)
                    print(i_a)
                    train_rmse = 0
                    train_mae = 0

                    loss_s = 0
                    accuracy_s = 0
                    mae_s = 0

                    ll_test = int(len(test_data) / batch_size) + 1
                    for batch_num in range(ll_test):
                        start_index = batch_num * batch_size
                        end_index = min((batch_num + 1) * batch_size, data_size_test)
                        data_test = test_data[start_index:end_index]

                        userid_valid, itemid_valid, reuid, reiid, y_valid = zip(*data_test)
                        u_valid = []
                        i_valid = []
                        for i in range(len(userid_valid)):
                            u_valid.append(u_text[userid_valid[i][0]])
                            i_valid.append(i_text[itemid_valid[i][0]])
                        u_valid = np.array(u_valid)
                        i_valid = np.array(i_valid)

                        loss, accuracy, mae = dev_step(u_valid, i_valid, userid_valid, itemid_valid, reuid, reiid,
                                                       y_valid)
                        loss_s = loss_s + len(u_valid) * loss
                        accuracy_s = accuracy_s + len(u_valid) * np.square(accuracy)
                        mae_s = mae_s + len(u_valid) * mae
                    print(f"loss_valid: {loss_s / test_length}, rmse_valid: {np.sqrt(accuracy_s / test_length)}, mae_valid: {mae_s / test_length}")
                    rmse = np.sqrt(accuracy_s / test_length)
                    mae = mae_s / test_length
                    if best_rmse > rmse:
                        best_rmse = rmse
                    if best_mae > mae:
                        best_mae = mae
                    print("")
                print('best rmse:' + best_rmse)
                print('best mae:' + best_mae)
