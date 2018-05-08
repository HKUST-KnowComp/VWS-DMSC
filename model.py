import tensorflow as tf
from tf.contrib.rnn import BasicLSTMCell
from func import dense, iterAttention


class model:
    def __init__(self, config, batch, word_mat, asp_word_mat, query_mat, trainable=True):
        self.config = config
        self.global_step = tf.get_variable("global_step", shape=[
        ], dtype=tf.int32, initializer=tf.constant_initializer(0), trainable=False)
        self.x, self.y, self.ay, self.w_mask, self.w_len, self.sent_num, self.asp, self.senti, self.weight, self.neg_senti = batch.get_next()
        self.is_train = tf.get_variable(
            "is_train", shape=[], dtype=tf.bool, trainable=False)
        self.word_mat = tf.get_variable(
            "word_mat", initializer=tf.constant(word_mat, dtype=tf.float32))
        self.asp_word_mat = tf.get_variable(
            "asp_word_mat", initializer=tf.constant(asp_word_mat, dtype=tf.float32))
        self.query_mat = tf.get_variable(
            "query_mat", initializer=tf.constant(query_mat, dtype=tf.float32))
        self.ready()

    def ready(self):
        config = self.config
        x, y, ay, w_mask, w_len, sent_num, asp, senti, weight, neg_senti = self.x, self.y, self.ay, self.w_mask, self.w_len, self.sent_num, self.asp, self.senti, self.weight, self.neg_senti
        word_mat, asp_word_mat, query_mat = self.word_mat, self.asp_word_mat, self.query_mat

        with tf.get_variable("encoder"):
            x = tf.nn.embedding_lookup(word_mat, x)
            cell_fw = BasicLSTMCell(config.hidden / 2)
            cell_bw = BasicLSTMCell(config.hidden / 2)
            x, _ = tf.nn.bidirectional_dynamic_rnn(
                cell_fw, cell_bw, x, sequence_length=w_len)
            query = tf.tanh(dense(query_mat, config.hidden))

            att = iterAttention(
                query, x, None, config.hidden, hop=config.hop_word)
