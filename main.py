import tensorflow as tf
from model import Model
from util.batch_gen import create_batch_generator
from util.load import load_corpus, load_query, load_embedding
from tqdm import tqdm


def train(config):
    word2idx, emb = load_embedding(config, config.emb)
    asp_word2idx, asp_emb = load_embedding(config, config.asp_emb)
    query_emb = load_query(config, config.aspect_seeds, word2idx, emb)

    input_types = (tf.int32, tf.float32, tf.float32, tf.float32, tf.int32,
                   tf.int32, tf.int32, tf.int32, tf.float32, tf.int32)
    input_shapes = (tf.TensorShape([None, None]), tf.TensorShape([None, None]), tf.TensorShape([None, None, None]), tf.TensorShape([None, None]), tf.TensorShape(
        [None]), tf.TensorShape([]), tf.TensorShape([None, None]), tf.TensorShape([None, None]), tf.TensorShape([None, None]), tf.TensorShape([None, None]))

    train_batch = tf.data.Dataset.from_generator(create_batch_generator(config, load_corpus(
        config, config.train, word2idx, asp_word2idx, filter_null=True)), input_types, input_shapes).repeat().shuffle(config.cache_size).make_one_shot_iterator()
    dev_batch = tf.data.Dataset.from_generator(create_batch_generator(config, load_corpus(
        config, config.dev, word2idx, asp_word2idx)), input_types, input_shapes).make_one_shot_iterator()
    test_batch = tf.data.Dataset.from_generator(create_batch_generator(config, load_corpus(
        config, config.test, word2idx, asp_word2idx)), input_types, input_shapes).make_one_shot_iterator()

    handle = tf.placeholder(tf.string, shape=[])
    batch = tf.data.Iterator.from_string_handle(
        handle, train_batch.output_types, train_batch.output_shapes)
    model = Model(config, batch, emb, asp_emb, query_emb)

    sess_config = tf.ConfigProto(allow_soft_placement=True)
    sess_config.gpu_options.allow_growth = True

    with tf.Session(config=sess_config) as sess:
        writer = tf.summary.FileWriter(config.log_dir)
        sess.run(tf.global_variables_initializer())
        train_handle = sess.run(train_batch.string_handle())
        dev_handle = sess.run(dev_batch.string_handle())
        test_handle = sess.run(test_batch.string_handle())
        for _ in tqdm(range(1, config.num_steps + 1), ascii=True):
            global_step = sess.run(model.global_step) + 1
            loss, _ = sess.run([model.loss, model.train_op],
                               feed_dict={handle: train_handle})
            if global_step % config.record_period == 0:
                loss_sum = tf.Summary(value=[tf.Summary.Value(
                    tag="model/loss", simple_value=loss), ])
                writer.add_summary(loss_sum, global_step)
                writer.flush()
