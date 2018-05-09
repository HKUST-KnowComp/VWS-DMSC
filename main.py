import os
import random
import tensorflow as tf
import numpy as np
from model import Model
from util.batch_gen import batch_generator, list_wrapper
from util.load import load_corpus, load_query, load_embedding
from tqdm import tqdm


def train(config):
    word2idx, emb = load_embedding(config, config.emb)
    asp_word2idx, asp_emb = load_embedding(config, config.asp_emb)
    query_emb = load_query(config, config.aspect_seeds, word2idx, emb)

    if config.overall and not config.unsupervised:
        query_emb = np.reshape(query_emb, [1, -1, config.emb_dim])
        config.num_aspects = 1

    if config.unsupervised:
        query_emb = np.asarray([query_emb[config.aspect]])
        config.num_aspects = 1

    print("Building Batches")
    train_batch_list = list(batch_generator(config, load_corpus(
        config, config.train, word2idx, asp_word2idx, filter_null=config.unsupervised)))
    dev_batch_list = list(batch_generator(config, load_corpus(
        config, config.dev, word2idx, asp_word2idx)))

    random.shuffle(train_batch_list)
    random.shuffle(dev_batch_list)
    num_train_batch = len(train_batch_list)
    num_dev_batch = len(dev_batch_list)

    input_types = (tf.int32, tf.float32, tf.float32, tf.float32, tf.int32,
                   tf.int32, tf.int32, tf.int32, tf.float32, tf.int32)
    input_shapes = (tf.TensorShape([None, None]), tf.TensorShape([None, None, None]), tf.TensorShape([None, None, None]), tf.TensorShape([None, None]), tf.TensorShape(
        [None]), tf.TensorShape([]), tf.TensorShape([None, None]), tf.TensorShape([None, None]), tf.TensorShape([None, None]), tf.TensorShape([None, None]))

    train_batch = tf.data.Dataset.from_generator(list_wrapper(
        train_batch_list), input_types, input_shapes).repeat().shuffle(config.cache_size).make_one_shot_iterator()
    dev_batch = tf.data.Dataset.from_generator(list_wrapper(
        dev_batch_list), input_types, input_shapes).repeat().make_one_shot_iterator()

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
        saver = tf.train.Saver(var_list=model.var_to_save,
                               max_to_keep=config.max_to_keep)
        if config.unsupervised:
            saver.restore(sess, tf.train.latest_checkpoint(config.save_dir))
        train_op = model.r_train_op if config.unsupervised else model.train_op
        sess.run(tf.assign(model.is_train, tf.constant(True, dtype=tf.bool)))
        for _ in tqdm(range(1, num_train_batch * config.num_epochs + 1), ascii=True):
            global_step = sess.run(model.global_step) + 1
            loss, _ = sess.run([model.loss, train_op],
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
                    config, model, config.num_batches, sess, handle, train_handle, tag="train")
                for s in summ:
                    writer.add_summary(s, global_step)
                _, _, summ = evaluate(
                    config, model, num_dev_batch, sess, handle, dev_handle, tag="dev")
                sess.run(tf.assign(model.is_train,
                                   tf.constant(True, dtype=tf.bool)))
                for s in summ:
                    writer.add_summary(s, global_step)
                writer.flush()
                filename = os.path.join(
                    config.save_dir, "model_{}.ckpt".format(global_step))
                saver.save(sess, filename)


def evaluate(config, model, num_batches, sess, handle, str_handle, tag="train"):
    mean_loss = 0.
    corr = [0. for _ in range(config.num_aspects)]
    total = [0. for _ in range(config.num_aspects)]
    overall_corr = 0.
    overall_total = 0.
    for _ in range(num_batches):
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
