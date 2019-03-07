import numpy as np
import tensorflow as tf
from itertools import product
from munkres import Munkres


class Evaluator:
    def __init__(self):
        self.last_round = None

    def __call__(self, config, model, num_batches, sess, handle, str_handle, tag="train", flip=False):
        num_aspects = config.num_aspects
        scale = config.score_scale

        mean_loss = 0.
        goldens = []
        preds = []
        for _ in range(num_batches):
            loss, pred, ay = sess.run([model.t_loss, model.pred, model.golden], feed_dict={handle: str_handle})
            mean_loss += loss
            golden = np.asarray([[np.argmax(col) if any([k > 0 for k in col]) else -
                                  1 for col in ay[i]] for i in range(num_aspects)], dtype=np.int32)
            goldens.append(golden)
            preds.append(pred)
        golden = np.concatenate(goldens, axis=1)
        pred = np.concatenate(preds, axis=1)
        mean_loss = mean_loss / config.num_batches

        if config.unsupervised:
            m = Munkres()
            aspect = config.aspect
            scale = config.score_scale
            tots = (golden[aspect] != -1).sum().astype(np.float32)
            confusion_mat = np.zeros([scale, scale], dtype=np.int32)
            for j, k in product(range(scale), range(scale)):
                confusion_mat[j, k] = - (np.logical_and(golden[aspect] == j, pred[0] == k).sum())
            idxs = m.compute(confusion_mat.tolist())
            cors = 0.
            for r, c in idxs:
                cors -= confusion_mat[r][c]
            cors = np.asarray(cors, dtype=np.float32)

        else:
            tots = (golden != -1).sum(axis=1).astype(np.float32)
            cors = (golden == pred).sum(axis=1).astype(np.float32)

        accs = (cors / tots).tolist()
        overall_acc = cors.sum() / tots.sum()

        summ = []
        loss_sum = tf.Summary(value=[tf.Summary.Value(tag="{}/loss".format(tag), simple_value=mean_loss), ])
        overall_acc_sum = tf.Summary(value=[tf.Summary.Value(tag="{}/acc".format(tag), simple_value=overall_acc)])
        summ.append(loss_sum)
        summ.append(overall_acc_sum)
        if not config.unsupervised and not config.overall:
            for i, acc in enumerate(accs):
                acc_sum = tf.Summary(value=[tf.Summary.Value(tag="{}/{}".format(tag, config.name_aspects[i]), simple_value=acc)])
                summ.append(acc_sum)
        if flip and config.unsupervised:
            self.last_round = pred if self.last_round is None else self.last_round
            flip = (pred != self.last_round).sum()
            self.last_round = pred
            flip_summ = tf.Summary(value=[tf.Summary.Value(tag="{}/flip".format(tag), simple_value=flip)])
            summ.append(flip_summ)
        return mean_loss, overall_acc, summ
