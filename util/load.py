import numpy as np
from util.word_dict import WordDict
from collections import defaultdict


def load_query(config, path, word2idx_dict, emb):
    mat = []
    with open(path, "r", encoding="iso-8859-1") as fh:
        for line in fh:
            mat.append([emb[word2idx_dict[seed]] for seed in line.strip().split()])
    return np.array(mat, dtype=np.float32)


def load_corpus(config, path, embedding, asp_embedding, filter_null=False):
    aspect = config.aspect + 1

    with open(path, "r", encoding="iso-8859-1") as fh:
        lines = fh.readlines()

    segs = [line.strip().split('\t\t\t') for line in lines]

    tmp_x = [seg[2].split('<ssssss>') for seg in segs]
    tmp_x = list(map(lambda doc: filter(lambda sent: sent, doc), tmp_x))
    tmp_asp = [seg[1].split('\t\t') for seg in segs]
    tmp_asp = list(map(lambda doc: filter(lambda asp: asp, doc), tmp_asp))
    asp_senti = list(map(lambda doc: list(map(lambda asp: asp.strip().split('\t'), doc)), tmp_asp))

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
                senti_ = asp_
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

    corpus_x = list(map(lambda doc: list(map(lambda sent: [embedding[word] for word in sent.strip().split()], doc)), tmp_x))
    corpus_y = list(map(lambda seg: list(map(lambda rating: int(rating) - 1, seg[0].strip().split())), segs))

    if filter_null:
        corpus_x = [corpus_x[i] for i, v in enumerate(valid) if v is True]
        corpus_y = [corpus_y[i] for i, v in enumerate(valid) if v is True]
        asp = [asp[i] for i, v in enumerate(valid) if v is True]
        senti = [senti[i] for i, v in enumerate(valid) if v is True]
        weight = [weight[i] for i, v in enumerate(valid) if v is True]
    return corpus_x, corpus_y, asp, senti, weight, senti_words


def load_embedding(config, path):
    word2idx_dict = defaultdict(int)
    embedding = [[0. for _ in range(config.emb_dim)]]
    with open(path, "r", encoding="iso-8859-1") as fh:
        for i, line in enumerate(fh, 1):
            line = line.split()
            word = " ".join(line[:-config.emb_dim])
            word2idx_dict[word] = i
            vec = list(map(float, line[-config.emb_dim:]))
            embedding.append(vec)
    return word2idx_dict, np.array(embedding, dtype=np.float32)
