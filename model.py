import tensorflow as tf
from tensorflow.contrib.rnn import BasicLSTMCell
from func import dense, iterAttention


class Model:
    def __init__(self, config, batch, word_mat, asp_word_mat, query_mat, trainable=True):
        self.config = config
        self.global_step = tf.get_variable("global_step", shape=[
        ], dtype=tf.int32, initializer=tf.constant_initializer(0), trainable=False)
        self.x, self.y, self.ay, self.w_mask, self.w_len, self.sent_num, self.asp, self.senti, self.weight, self.neg_senti = batch.get_next()
        self.num_aspect = query_mat.shape[0]
        self.is_train = tf.get_variable(
            "is_train", shape=[], dtype=tf.bool, trainable=False)
        self.word_mat = tf.get_variable(
            "word_mat", initializer=tf.constant(word_mat, dtype=tf.float32))
        self.asp_word_mat = tf.get_variable(
            "asp_word_mat", initializer=tf.constant(asp_word_mat, dtype=tf.float32))
        self.query_mat = tf.get_variable(
            "query_mat", initializer=tf.constant(query_mat, dtype=tf.float32))
        self.ready()
        if trainable:
            self.opt = tf.train.AdadeltaOptimizer(
                learning_rate=config.learning_rate, epsilon=1e-6)
            self.train_op = self.opt.minimize(
                self.loss, global_step=self.global_step)

    def ready(self):
        config = self.config
        x, y, ay, w_mask, w_len, num_sent, asp, senti, weight, neg_senti = self.x, self.y, self.ay, self.w_mask, self.w_len, self.sent_num, self.asp, self.senti, self.weight, self.neg_senti
        word_mat, asp_word_mat, query_mat = self.word_mat, self.asp_word_mat, self.query_mat

        num_aspect = self.num_aspect
        dim_sent = tf.shape(x)[0]
        batch = tf.floordiv(tf.shape(x)[0], num_sent)

        with tf.variable_scope("word_level"):
            x = tf.nn.embedding_lookup(word_mat, x)
            cell_fw = BasicLSTMCell(config.hidden / 2)
            cell_bw = BasicLSTMCell(config.hidden / 2)
            (x_fw, x_bw), _ = tf.nn.bidirectional_dynamic_rnn(
                cell_fw, cell_bw, x, sequence_length=w_len, dtype=tf.float32)
            x = tf.concat([x_fw, x_bw], axis=-1)
            query = tf.tanh(dense(query_mat, config.hidden))

            query = tf.tile(tf.expand_dims(query, axis=1), [1, dim_sent, 1, 1])
            doc = tf.tile(tf.expand_dims(x, axis=0), [num_aspect, 1, 1, 1])
            mask = tf.expand_dims(tf.tile(tf.expand_dims(
                w_mask, axis=0), [num_aspect, 1, 1]), axis=3)

            att = iterAttention(query, doc, mask, hop=config.hop_word)
            att = tf.reshape(
                att, [num_aspect * batch, num_sent, config.hidden * config.hop_word])

        with tf.variable_scope("sent_level"):
            cell2_fw = BasicLSTMCell(config.hidden / 2)
            cell2_bw = BasicLSTMCell(config.hidden / 2)
            (att_fw, att_bw), _ = tf.nn.bidirectional_dynamic_rnn(
                cell2_fw, cell2_bw, att, dtype=tf.float32)
            att = tf.concat([att_fw, att_bw], axis=-1)
            query = tf.tanh(dense(query_mat, config.hidden))

            query = tf.tile(tf.expand_dims(query, axis=1), [1, batch, 1, 1])
            doc = tf.reshape(att, [num_aspect, batch, num_sent, config.hidden])
            att = iterAttention(query, doc, hop=config.hop_sent)

        with tf.variable_scope("predict"):
            losses = []
            for i in range(num_aspect):
                with tf.variable_scope("aspect_{}".format(i)):
                    prob = tf.nn.softmax(
                        dense(att[i], config.score_scale), axis=1)
                    loss = tf.reduce_sum(-ay[i] * tf.log(prob + 1e-5), axis=1)
                    losses.append(tf.reduce_mean(loss))
            self.loss = tf.reduce_mean(losses)
