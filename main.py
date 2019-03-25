import os
import random
from itertools import chain
import tensorflow as tf
from model import Model
from util.batch_gen import batch_generator, list_wrapper
from util.load import load_corpus, load_query, load_embedding
from tqdm import tqdm
from evaluator import Evaluator


def train(config):
    word2idx, emb = load_embedding(config, config.emb)
    asp_word2idx, asp_emb = load_embedding(config, config.asp_emb)
    query_emb = load_query(config, config.aspect_seeds, word2idx, emb)

    if config.overall:
        config.num_aspects = 1

    print("Building Batches")
    train_batch_list = list(batch_generator(config, load_corpus(config, config.train, word2idx, asp_word2idx, filter_null=config.unsupervised)))
    dev_batch_list = list(batch_generator(config, load_corpus(config, config.dev, word2idx, asp_word2idx)))
    test_batch_list = list(batch_generator(config, load_corpus(config, config.test, word2idx, asp_word2idx)))

    random.shuffle(train_batch_list)
    random.shuffle(dev_batch_list)
    random.shuffle(test_batch_list)
    num_train_batch = len(train_batch_list)
    num_dev_batch = len(dev_batch_list)
    num_test_batch = len(test_batch_list)

    input_types = (tf.int32, tf.float32, tf.float32, tf.float32, tf.int32,
                   tf.int32, tf.int32, tf.int32, tf.float32, tf.int32)
    input_shapes = (tf.TensorShape([None, None]), tf.TensorShape([None, None, None]), tf.TensorShape([None, None, None]), tf.TensorShape([None, None]), tf.TensorShape(
        [None]), tf.TensorShape([]), tf.TensorShape([None, None]), tf.TensorShape([None, None]), tf.TensorShape([None, None]), tf.TensorShape([None, None]))

    train_batch = tf.data.Dataset.from_generator(list_wrapper(
        train_batch_list), input_types, input_shapes).repeat().shuffle(config.cache_size).make_one_shot_iterator()
    dev_batch = tf.data.Dataset.from_generator(list_wrapper(
        dev_batch_list), input_types, input_shapes).repeat().make_one_shot_iterator()
    test_batch = tf.data.Dataset.from_generator(list_wrapper(
        test_batch_list), input_types, input_shapes).repeat().make_one_shot_iterator()

    train_evaluator = Evaluator()
    dev_evaluator = Evaluator()
    test_evaluator = Evaluator()

    handle = tf.placeholder(tf.string, shape=[])
    batch = tf.data.Iterator.from_string_handle(handle, train_batch.output_types, train_batch.output_shapes)
    model = Model(config, batch, emb, asp_emb, query_emb)

    sess_config = tf.ConfigProto(allow_soft_placement=True)
    sess_config.gpu_options.allow_growth = True

    with tf.Session(config=sess_config) as sess:
        writer = tf.summary.FileWriter(config.log_dir)
        sess.run(tf.global_variables_initializer())
        train_handle = sess.run(train_batch.string_handle())
        dev_handle = sess.run(dev_batch.string_handle())
        test_handle = sess.run(test_batch.string_handle())
        saver = tf.train.Saver(var_list=model.var_to_save, max_to_keep=config.max_to_keep)
        if config.unsupervised:
            saver.restore(sess, tf.train.latest_checkpoint(config.save_dir))
        sess.run(tf.assign(model.is_train, tf.constant(True, dtype=tf.bool)))
        best_val_acc, best_test_acc = 0., 0.
        for _ in tqdm(range(1, num_train_batch * config.num_epochs + 1)):
            global_step = sess.run(model.global_step) + 1
            loss, _ = sess.run([model.t_loss, model.train_op], feed_dict={handle: train_handle})

            if global_step % config.record_period == 0:
                loss_sum = tf.Summary(value=[tf.Summary.Value(tag="model/loss", simple_value=loss), ])
                writer.add_summary(loss_sum, global_step)
                writer.flush()

            if global_step % config.eval_period == 0:
                sess.run(tf.assign(model.is_train, tf.constant(False, dtype=tf.bool)))
                _, _, train_summ = train_evaluator(config, model, config.num_batches, sess, handle, train_handle, tag="train")
                _, val_acc, dev_summ = dev_evaluator(config, model, num_dev_batch, sess, handle, dev_handle, tag="dev", flip=True)
                _, test_acc, test_summ = test_evaluator(config, model, num_test_batch, sess, handle, test_handle, tag="test", flip=True)
                for s in chain(train_summ, dev_summ, test_summ):
                    writer.add_summary(s, global_step)
                sess.run(tf.assign(model.is_train, tf.constant(True, dtype=tf.bool)))
                writer.flush()
                if val_acc > best_val_acc:
                    best_val_acc, best_test_acc = val_acc, test_acc
                    if not config.unsupervised:
                        filename = os.path.join(config.save_dir, "model_{}.ckpt".format(global_step))
                        saver.save(sess, filename)
        print("Dev Acc {}, Test Acc {}".format(best_val_acc, best_test_acc))
