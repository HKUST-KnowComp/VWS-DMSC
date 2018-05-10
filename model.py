import tensorflow as tf
from func import dense, iter_attention, dropout, cudnn_lstm


class Model:
    def __init__(self, config, batch, word_mat, asp_word_mat, query_mat):
        self.config = config
        self.global_step = tf.get_variable("global_step", shape=[
        ], dtype=tf.int32, initializer=tf.constant_initializer(0), trainable=False)
        self.x, self.y, self.ay, self.w_mask, self.w_len, self.sent_num, self.asp, self.senti, self.weight, self.neg_senti = batch.get_next()
        self.num_aspect = config.num_aspects
        self.is_train = tf.get_variable(
            "is_train", shape=[], dtype=tf.bool, trainable=False)
        self.word_mat = tf.get_variable(
            "word_mat", initializer=tf.constant(word_mat, dtype=tf.float32))
        self.asp_word_mat = asp_word_mat
        self.query_mat = tf.get_variable(
            "query_mat", initializer=tf.constant(query_mat, dtype=tf.float32))

        self.ready()

        self.word_level_vars = tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES, scope="word_level")
        self.sent_level_vars = tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES, scope="sent_level")
        self.predict_vars = tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES, scope="predict")
        self.decoder_vars = tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES, scope="decoder")
        self.var_to_save = self.word_level_vars + self.sent_level_vars

        en_reg = tf.contrib.layers.l2_regularizer(config.en_l2_reg)
        de_reg = tf.contrib.layers.l2_regularizer(config.de_l2_reg)
        en_l2 = tf.contrib.layers.apply_regularization(
            en_reg, self.word_level_vars + self.sent_level_vars)
        de_l2 = tf.contrib.layers.apply_regularization(
            de_reg, self.predict_vars + self.decoder_vars)
        self.l2_loss = en_l2 + de_l2

        self.opt = tf.train.AdadeltaOptimizer(
            learning_rate=config.learning_rate, epsilon=1e-6)
        self.r_opt = tf.train.AdadeltaOptimizer(
            learning_rate=config.learning_rate, epsilon=1e-6)
        self.train_op = self.opt.minimize(
            self.loss + self.l2_loss, global_step=self.global_step)
        self.r_train_op = self.r_opt.minimize(
            self.r_loss + self.l2_loss, var_list=self.predict_vars + self.decoder_vars, global_step=self.global_step)

    def ready(self):
        config = self.config
        x, y, ay, w_mask, w_len, num_sent, senti, weight, neg_senti = self.x, self.y, self.ay, self.w_mask, self.w_len, self.sent_num, self.senti, self.weight, self.neg_senti
        word_mat, asp_word_mat, query_mat = self.word_mat, self.asp_word_mat, self.query_mat

        target = y if config.overall else ay

        num_aspect = self.num_aspect
        score_scale = config.score_scale
        batch = tf.floordiv(tf.shape(x)[0], num_sent)

        with tf.variable_scope("word_level"):
            x = dropout(tf.nn.embedding_lookup(word_mat, x),
                        keep_prob=config.keep_prob, is_train=self.is_train)
            x = cudnn_lstm(x, config.hidden // 2, sequence_length=w_len)
            query = tf.tanh(dense(query_mat, config.hidden))

            doc = tf.expand_dims(x, axis=0)
            mask = tf.expand_dims(w_mask, axis=0)

            att = iter_attention(query, doc, mask, hop=config.hop_word)
            att = tf.reshape(
                att, [num_aspect * batch, num_sent, config.hidden * config.hop_word])

        with tf.variable_scope("sent_level"):
            att = dropout(att, keep_prob=config.keep_prob,
                          is_train=self.is_train)
            att = cudnn_lstm(att, config.hidden // 2)
            query = tf.tanh(dense(query_mat, config.hidden))

            doc = tf.reshape(att, [num_aspect, batch, num_sent, config.hidden])
            att = iter_attention(query, doc, hop=config.hop_sent)

        with tf.variable_scope("predict"):
            probs = []
            losses = []
            preds = []
            att = dropout(att, keep_prob=config.keep_prob,
                          is_train=self.is_train)
            for i in range(num_aspect):
                with tf.variable_scope("aspect_{}".format(i)):
                    prob = tf.nn.softmax(
                        dense(att[i], config.score_scale, use_bias=False))
                    loss = tf.reduce_sum(
                        -target[i] * tf.log(prob + 1e-5), axis=1)
                    probs.append(prob)
                    preds.append(tf.argmax(prob, axis=1))
                    losses.append(tf.reduce_mean(loss))
            self.probs = tf.stack(probs, axis=0)
            self.pred = tf.stack(preds, axis=0)
            self.loss = tf.reduce_mean(losses)

        with tf.variable_scope("decoder"):
            emb = tf.get_variable("emb", initializer=tf.constant(
                asp_word_mat, dtype=tf.float32))
            sent_emb = tf.nn.embedding_lookup(emb, senti)
            neg_sent_emb = tf.nn.embedding_lookup(emb, neg_senti)
            with tf.variable_scope("selectional_preference", reuse=tf.AUTO_REUSE):
                w = tf.expand_dims(weight, axis=2)
                u = dense(sent_emb, score_scale, use_bias=False)
                v = dense(neg_sent_emb, score_scale, use_bias=False)
                u = tf.reduce_sum(tf.log(tf.nn.softmax(u * w)), axis=1)
                v = tf.reduce_sum(tf.log(tf.nn.softmax(-v)), axis=1)

                r_loss = tf.reduce_sum(
                    (u + v) * self.probs[config.aspect], axis=1)
                u_loss = tf.reduce_sum(u * self.probs[config.aspect], axis=1)

                w = tf.reduce_max(tf.abs(weight), axis=1)
                num = tf.reduce_sum(w) + 1e-5
            self.r_loss = r_loss / num
            self.u_loss = u_loss / num
