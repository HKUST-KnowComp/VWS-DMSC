import tensorflow as tf
from func import dense, iter_attention, dropout, cudnn_lstm, selectional_preference


class Model:
    def __init__(self, config, batch, word_mat, asp_word_mat, query_mat):
        self.config = config
        self.global_step = tf.get_variable("global_step", shape=[], dtype=tf.int32, initializer=tf.constant_initializer(0), trainable=False)
        self.x, self.y, self.ay, self.w_mask, self.w_len, self.sent_num, self.asp, self.senti, self.weight, self.neg_senti = batch.get_next()
        self.num_aspect = config.num_aspects
        self.is_train = tf.get_variable("is_train", shape=[], dtype=tf.bool, trainable=False)
        self.word_mat = tf.get_variable("word_mat", initializer=tf.constant(word_mat, dtype=tf.float32), trainable=False)
        self.asp_word_mat = tf.get_variable("asp_word_mat", initializer=tf.constant(asp_word_mat, dtype=tf.float32))
        self.query_mat = tf.get_variable("query_mat", initializer=tf.constant(query_mat, dtype=tf.float32))
        self.loss, self.r_loss, self.u_loss = None, None, None

        self.ready()

        word_level_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="word_level")
        sent_level_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="sent_level")
        pred_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="predict")
        dec_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="decoder")
        self.var_to_save = word_level_vars + sent_level_vars + [self.query_mat]

        sup_vars = word_level_vars + sent_level_vars + pred_vars + [self.query_mat]
        un_vars = pred_vars + dec_vars

        en_reg = tf.contrib.layers.l2_regularizer(config.en_l2_reg)
        de_reg = tf.contrib.layers.l2_regularizer(config.de_l2_reg)
        sup_l2 = tf.contrib.layers.apply_regularization(en_reg, sup_vars)
        un_l2 = tf.contrib.layers.apply_regularization(de_reg, un_vars)

        self.t_loss = self.r_loss + un_l2 if config.unsupervised else self.loss + sup_l2
        var_list = un_vars + [self.asp_word_mat] if config.unsupervised else None
        self.opt = tf.train.AdadeltaOptimizer(config.learning_rate)
        self.train_op = self.opt.minimize(self.t_loss, global_step=self.global_step, var_list=var_list)

    def ready(self):
        config = self.config
        x, w_mask, w_len, num_sent, senti, weight, neg_senti = self.x, self.w_mask, self.w_len, self.sent_num, self.senti, self.weight, self.neg_senti
        word_mat, asp_word_mat, query_mat = self.word_mat, self.asp_word_mat, self.query_mat

        num_aspect = self.num_aspect
        score_scale = config.score_scale
        batch = tf.floordiv(tf.shape(x)[0], num_sent)
        query_mat = tf.reshape(query_mat, [config.num_aspects, -1, config.emb_dim])

        with tf.variable_scope("word_level"):
            x = dropout(tf.nn.embedding_lookup(word_mat, x), keep_prob=config.keep_prob, is_train=self.is_train)
            x = cudnn_lstm(x, config.hidden // 2, sequence_length=w_len)
            query = tf.tanh(dense(query_mat, config.hidden))

            doc = tf.expand_dims(x, axis=0)
            mask = tf.expand_dims(w_mask, axis=0)

            att = iter_attention(query, doc, mask, hop=config.hop_word)
            att = tf.reshape(att, [num_aspect * batch, num_sent, config.hidden * config.hop_word])

        with tf.variable_scope("sent_level"):
            att = dropout(att, keep_prob=config.keep_prob, is_train=self.is_train)
            att = cudnn_lstm(att, config.hidden // 2)
            query = tf.tanh(dense(query_mat, config.hidden))

            doc = tf.reshape(att, [num_aspect, batch, num_sent, config.hidden])
            att = iter_attention(query, doc, hop=config.hop_sent)

        with tf.variable_scope("predict"):
            probs = []
            att = dropout(att, keep_prob=config.keep_prob, is_train=self.is_train)
            aspects = [config.aspect] if config.unsupervised else list(range(num_aspect))
            for i in aspects:
                with tf.variable_scope("aspect_{}".format(i)):
                    probs.append(tf.nn.softmax(dense(att[i], score_scale)))
            self.prob = tf.stack(probs, axis=0)
            self.pred = tf.argmax(self.prob, axis=2)

            self.golden = self.y if config.overall else self.ay
            self.loss = tf.reduce_sum(tf.reduce_mean(tf.reduce_sum(-self.golden * tf.log(self.prob + 1e-6), axis=2), axis=1))

        with tf.variable_scope("decoder"):
            sent_emb = tf.nn.embedding_lookup(asp_word_mat, senti)
            neg_sent_emb = tf.nn.embedding_lookup(asp_word_mat, neg_senti)
            self.r_loss, self.u_loss = selectional_preference(sent_emb, neg_sent_emb, weight, self.prob[0], score_scale, alpha=config.alpha)
