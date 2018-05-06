import tensorflow as tf
from util.batch_gen import load_corpus, load_embedding, create_batch_generator


def train(config):
    word2idx, emb = load_embedding(config, config.emb)
    asp_word2idx, asp_emb = load_embedding(config, config.asp_emb)

    input_types = (tf.int32, tf.int32, tf.int32, tf.float32, tf.int32,
                   tf.int32, tf.int32, tf.int32, tf.float32, tf.int32)

    train_batch = tf.data.Dataset.from_generator(create_batch_generator(config, load_corpus(
        config, config.train, word2idx, asp_word2idx, filter_null=True)), input_types).repeat().shuffle(config.cache_size).make_one_shot_iterator()
    dev_batch = tf.data.Dataset.from_generator(create_batch_generator(config, load_corpus(
        config, config.dev, word2idx, asp_word2idx)), input_types).make_one_shot_iterator()
    test_batch = tf.data.Dataset.from_generator(create_batch_generator(config, load_corpus(
        config, config.test, word2idx, asp_word2idx)), input_types).make_one_shot_iterator()
