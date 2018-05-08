import numpy as np
import random


def create_one_batch(arg, ids, corpus):
    x, y, asp, senti, weight, senti_words = corpus
    max_len = 0
    for i in ids:
        for sent in x[i]:
            max_len = max(max_len, len(sent))

    batch_x = [np.asarray([sent + [-1] * (max_len - len(sent))
                           for sent in x[i]], dtype=np.int32) for i in ids]
    batch_w_mask = [np.asarray([[1.] * len(sent) + [0.] * (max_len - len(sent))
                                for sent in x[i]], dtype=np.float32) for i in ids]
    batch_w_len = [np.asarray([len(sent)
                               for sent in x[i]], dtype=np.int32) for i in ids]

    batch_x = np.concatenate(batch_x, axis=0)
    batch_w_mask = np.concatenate(batch_w_mask, axis=0)
    batch_w_len = np.concatenate(batch_w_len, axis=0)

    batch_y = np.array([y[i][0] for i in ids])
    batch_ay = np.array([y[i][1:] for i in ids])
    batch_asp = []
    batch_senti = []
    batch_weight = []
    batch_neg_senti = []

    for i in ids:
        neg_senti_ = []
        senti_count = senti_words.count
        words = []
        p = []

        for idx, word in enumerate(senti[i]):
            if word in senti_count and senti_count[word] >= arg.min_count:
                words.append(idx)
                p.append(senti_count[word] ** -0.25)

        if len(p) > 0:
            total = sum(p)
            p = [k / total for k in p]
            ran_val = np.random.choice(words, arg.num_senti, p=p)
            asp_ = [asp[i][val] for val in ran_val]
            senti_ = [senti[i][val] for val in ran_val]
            weight_ = [weight[i][val] for val in ran_val]
            neg_senti_ = [senti_words.sample(
                min_count=arg.min_count) for _ in range(arg.neg_num)]

        else:
            asp_ = [0 for _ in range(arg.num_senti)]
            senti_ = [0 for _ in range(arg.num_senti)]
            weight_ = [0. for _ in range(arg.num_senti)]
            neg_senti_ = [0 for _ in range(arg.neg_num)]

        batch_asp.append(asp_)
        batch_senti.append(senti_)
        batch_weight.append(weight_)
        batch_neg_senti.append(neg_senti_)

    batch_asp = np.transpose(np.asarray(batch_asp), (1, 0))
    batch_senti = np.transpose(np.asarray(batch_senti), (1, 0))
    batch_weight = np.transpose(np.asarray(batch_weight), (1, 0))
    batch_neg_senti = np.transpose(np.asarray(batch_neg_senti), (1, 0))
    return batch_x, batch_y, batch_ay, batch_w_mask, batch_w_len, batch_asp, batch_senti, batch_weight, batch_neg_senti


def create_batch_generator(arg, corpus):
    def create_batches():
        batch_size = arg.batch_size
        idxs = list(range(len(corpus[0])))
        random.shuffle(idxs)
        idxs = sorted(idxs, key=lambda i: len(corpus[0][i]))
        batch_idx = [idxs[0]]
        for i in idxs:
            if len(batch_idx) < batch_size and len(corpus[0][batch_idx[0]]) == len(corpus[0][i]):
                batch_idx.append(i)
            else:
                batch_x, batch_y, batch_ay, batch_w_mask, batch_w_len, batch_asp, batch_senti, batch_weight, batch_neg_senti = create_one_batch(
                    arg, batch_idx, corpus)
                sent_num = len(corpus[0][batch_idx[0]])
                batch_idx = [i]
                yield batch_x, batch_y, batch_ay, batch_w_mask, batch_w_len, sent_num, batch_asp, batch_senti, batch_weight, batch_neg_senti
    return create_batches
