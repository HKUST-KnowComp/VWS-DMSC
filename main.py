import os
import tensorflow as tf
import numpy as np
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
        config, config.dev, word2idx, asp_word2idx)), input_types, input_shapes).repeat().make_one_shot_iterator()
    test_batch = tf.data.Dataset.from_generator(create_batch_generator(config, load_corpus(
        config, config.test, word2idx, asp_word2idx)), input_types, input_shapes).repeat().make_one_shot_iterator()

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
        saver = tf.train.Saver()
        sess.run(tf.assign(model.is_train, tf.constant(True, dtype=tf.bool)))
        for _ in tqdm(range(1, config.num_steps + 1), ascii=True):
            global_step = sess.run(model.global_step) + 1
            loss, _ = sess.run([model.loss, model.train_op],
                               feed_dict={handle: train_handle})

            if global_step % config.record_period == 0:
                loss_sum = tf.Summary(value=[tf.Summary.Value(
                    tag="model/loss", simple_value=loss), ])
                writer.add_summary(loss_sum, global_step)
                writer.flush()

            if global_step % config.eval_period == 0:
                sess.run(tf.assign(model.is_train,
                                   tf.constant(False, dtype=tf.bool)))
                _, _, summ = evaluate(
                    config, model, sess, handle, train_handle, tag="train")
                for s in summ:
                    writer.add_summary(s, global_step)
                _, _, summ = evaluate(
                    config, model, sess, handle, dev_handle, tag="dev")
                sess.run(tf.assign(model.is_train,
                                   tf.constant(True, dtype=tf.bool)))
                for s in summ:
                    writer.add_summary(s, global_step)
                writer.flush()
                filename = os.path.join(
                    config.save_dir, "model_{}.ckpt".format(global_step))
                saver.save(sess, filename)


def evaluate(config, model, sess, handle, str_handle, tag="train"):
    mean_loss = 0.
    corr = [0. for _ in range(config.num_aspects)]
    total = [0. for _ in range(config.num_aspects)]
    overall_corr = 0.
    overall_total = 0.
    for _ in range(config.num_batches):
        loss, pred, ay = sess.run(
            [model.loss, model.pred, model.ay], feed_dict={handle: str_handle})
        mean_loss += loss
        for i in range(config.num_aspects):
            for j in range(ay[i].shape[0]):
                if any([k > 0 for k in ay[i][j]]):
                    gt = np.argmax(ay[i][j])
                    total[i] += 1.
                    overall_total += 1.
                    if gt == pred[i][j]:
                        corr[i] += 1.
                        overall_corr += 1.
    mean_loss = mean_loss / config.num_batches
    acc = [cor / tot for cor, tot in zip(corr, total)]
    overall_acc = overall_corr / overall_total
    summ = []
    loss_sum = tf.Summary(value=[tf.Summary.Value(
        tag="{}/loss".format(tag), simple_value=mean_loss), ])
    overall_acc_sum = tf.Summary(
        value=[tf.Summary.Value(tag="{}/acc".format(tag), simple_value=overall_acc)])
    summ.append(loss_sum)
    summ.append(overall_acc_sum)
    for i, ac in enumerate(acc):
        acc_sum = tf.Summary(value=[tf.Summary.Value(
            tag="{}/{}".format(tag, config.name_aspects[i]), simple_value=ac)])
        summ.append(acc_sum)
    return mean_loss, acc, summ
