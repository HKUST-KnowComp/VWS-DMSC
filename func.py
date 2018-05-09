import tensorflow as tf

INF = 1e20


def cudnn_lstm(inputs, num_units, sequence_length=None, scope="cudnn_lstm"):
    with tf.variable_scope(scope):
        inputs_fw = tf.transpose(inputs, [1, 0, 2])
        cell_fw = tf.contrib.cudnn_rnn.CudnnLSTM(1, num_units)
        cell_bw = tf.contrib.cudnn_rnn.CudnnLSTM(1, num_units)
        out_fw, _ = cell_fw(inputs_fw)
        if sequence_length is not None:
            inputs_bw = tf.reverse_sequence(
                inputs_fw, seq_lengths=sequence_length, seq_dim=0, batch_dim=1)
        else:
            inputs_bw = tf.reverse(inputs_fw, axis=[0])
        out_bw, _ = cell_bw(inputs_bw)
        if sequence_length is not None:
            out_bw = tf.reverse_sequence(
                out_bw, seq_lengths=sequence_length, seq_dim=0, batch_dim=1)
        else:
            out_bw = tf.reverse(out_bw, axis=[0])
        out = tf.transpose(tf.concat([out_fw, out_bw], axis=2), [1, 0, 2])
        return out


def dense(inputs, hidden, use_bias=True, scope="dense"):
    with tf.variable_scope(scope):
        shape = tf.shape(inputs)
        dim = inputs.get_shape().as_list()[-1]
        out_shape = [shape[idx] for idx in range(
            len(inputs.get_shape().as_list()) - 1)] + [hidden]
        flat_inputs = tf.reshape(inputs, [-1, dim])
        W = tf.get_variable("W", [dim, hidden])
        res = tf.matmul(flat_inputs, W)
        if use_bias:
            b = tf.get_variable(
                "b", [hidden], initializer=tf.constant_initializer(0.))
            res = tf.nn.bias_add(res, b)
        res = tf.reshape(res, out_shape)
        return res


def dropout(args, keep_prob, is_train, mode=None):
    if keep_prob < 1.0:
        noise_shape = None
        scale = 1.0
        shape = tf.shape(args)
        if mode == "recurrent" and len(args.get_shape().as_list()) == 3:
            noise_shape = [shape[0], 1, shape[-1]]
        args = tf.cond(is_train, lambda: tf.nn.dropout(
            args, keep_prob, noise_shape=noise_shape) * scale, lambda: args)
    return args


def softmax_mask(val, mask):
    return -INF * (1 - tf.cast(mask, tf.float32)) + val


def iterAttention(query, doc, mask=None, hop=1, scope="iter"):
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        num_aspect = tf.shape(query)[0]
        dim_sent = tf.shape(doc)[1]
        dim = query.get_shape().as_list()[-1]
        att = tf.tile(tf.zeros([1, 1, 1]), [num_aspect, dim_sent, dim])
        res = []
        for _ in range(hop):
            with tf.variable_scope("query"):
                att = dense(att, dim, use_bias=False, scope="pick")
                alpha = tf.tanh(query * tf.expand_dims(att, axis=2))
                alpha = tf.nn.softmax(
                    dense(alpha, 1, use_bias=False, scope="query"), axis=2)
                att = tf.reduce_sum(alpha * query, axis=2)

            with tf.variable_scope("doc"):
                att = dense(att, dim, use_bias=False, scope="pick")
                alpha = tf.tanh(doc * tf.expand_dims(att, axis=2))
                alpha = dense(alpha, 1, use_bias=False, scope="doc")
                if mask is not None:
                    alpha = softmax_mask(alpha, mask)
                alpha = tf.nn.softmax(alpha, axis=1)
                att = tf.reduce_sum(alpha * doc, axis=2)
            res.append(att + 0.)
        return tf.concat(res, axis=-1)
