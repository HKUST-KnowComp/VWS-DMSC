import numpy as np
import random
from util.word_dict import WordDict
from collections import defaultdict
from tqdm import tqdm
from functools import reduce


def load_corpus(config, path, embedding, asp_embedding, filter_null=False):
    aspect = config.aspect

    with open(path, "r") as fh:
        lines = fh.readlines()

    segs = [line.strip().split('\t\t\t') for line in lines]

    tmp_x = [seg[2].split('<ssssss>') for seg in segs]
    tmp_x = list(map(lambda doc: filter(lambda sent: sent, doc), tmp_x))
    tmp_asp = [seg[1].split('\t\t') for seg in segs]
    tmp_asp = list(map(lambda doc: filter(lambda asp: asp, doc), tmp_asp))
    asp_senti = list(map(lambda doc: list(
        map(lambda asp: asp.strip().split('\t'), doc)), tmp_asp))

    asp = []
    senti = []
    weight = []
    valid = []
    senti_words = WordDict()

    for idx, sample in enumerate(asp_senti):
        sample_asp = []
        sample_weight = []
        sample_senti = []
        sample_valid = False
        for i in range(len(sample[aspect]) // 2):
            asp_ = sample[aspect][2 * i]
            senti_ = sample[aspect][2 * i + 1]
            wei_ = 1.
            if " no" in senti:
                senti_ = senti.split()[0].strip()
                wei_ = -1.
            if senti == "no":
                senti_ = aspect
                wei_ = -1.
            if asp_ in asp_embedding and senti_ in asp_embedding:
                sample_asp.append(asp_embedding[asp_])
                sample_senti.append(asp_embedding[senti_])
                senti_words.add(asp_embedding[senti_])
                sample_weight.append(wei_)
                sample_valid = True
        asp.append(sample_asp)
        senti.append(sample_senti)
        weight.append(sample_weight)
        valid.append(sample_valid)

    corpus_x = list(map(lambda doc: list(map(
        lambda sent: [embedding[word] for word in sent.strip().split()], doc)), tmp_x))
    corpus_y = list(map(lambda seg: list(map(lambda rating: int(
        rating) - 1, seg[0].strip().split())), segs))

    if filter_null:
        corpus_x = [corpus_x[i] for i, v in enumerate(valid) if v is True]
        corpus_y = [corpus_y[i] for i, v in enumerate(valid) if v is True]
        asp = [asp[i] for i, v in enumerate(valid) if v is True]
        senti = [senti[i] for i, v in enumerate(valid) if v is True]
        weight = [weight[i] for i, v in enumerate(valid) if v is True]
    return corpus_x, corpus_y, asp, senti, weight, senti_words


def load_embedding(config, path):
    word2idx_dict = defaultdict(int)
    embedding = []
    with open(path, "r", encoding="utf-8") as fh:
        for i, line in enumerate(fh):
            line = line.split()
            word = " ".join(line[:-config.emb_dim])
            word2idx_dict[word] = i
            vec = map(float, word[-config.emb_dim:])
            embedding.append(vec)
    return word2idx_dict, embedding


def create_one_batch(arg, ids, corpus):
    x, y, asp, senti, weight, senti_words = corpus
    max_len = 0
    for iid in ids:
        for sent in x[iid]:
            max_len = max(max_len, len(sent))
    batch_x = list(map(lambda iid: np.asarray(list(map(
        lambda sent: sent + [-1] * (max_len - len(sent)), x[iid])), dtype=np.int32).T, ids))
    batch_w_mask = list(map(lambda iid: np.asarray(list(map(lambda sent: len(
        sent) * [1] + [0] * (max_len - len(sent)), x[iid])), dtype=np.float32).T, ids))
    batch_w_len = list(map(lambda iid: np.asarray(list(map(lambda sent: len(
        sent), x[iid])), dtype=np.float32) + np.float32(1e-4), ids))

    batch_x = reduce(lambda doc, docs: np.concatenate(
        (doc, docs), axis=1), batch_x)
    batch_w_mask = reduce(lambda doc, docs: np.concatenate(
        (doc, docs), axis=1), batch_w_mask)
    batch_w_len = reduce(lambda doc, docs: np.concatenate(
        (doc, docs), axis=0), batch_w_len)

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
        length = len(corpus[0])
        idxs = list(range(length))
        random.shuffle(idxs)
        idxs = sorted(idxs, key=lambda i: len(corpus[0][i]))
        batch_idx = [idxs[0]]
        for i in tqdm(range(length), ascii=True):
            if len(batch_idx) < batch_size and len(corpus[0][batch_idx[0]]) == len(corpus[0][i]):
                batch_idx.append(i)
            else:
                batch_x, batch_y, batch_ay, batch_w_mask, batch_w_len, batch_asp, batch_senti, batch_weight, batch_neg_senti = create_one_batch(
                    arg, batch_idx, corpus)
                sent_num = len(corpus[0][batch_idx[0]])
                batch_idx = [i]
                yield batch_x, batch_y, batch_ay, batch_w_mask, batch_w_len, sent_num, batch_asp, batch_senti, batch_weight, batch_neg_senti
    return create_batches
