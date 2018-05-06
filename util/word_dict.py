import numpy as np
import math


class WordDict:

    def __init__(self):

        self.count = {}
        self.if_ready = False
        self.words = None
        self.probs = None

    def add(self, word):
        if word not in self.count:
            self.count[word] = 0
        self.count[word] += 1
        self.if_ready = False

    def ready(self, min_count=None):
        total = 0.
        words = []
        probs = []
        for key, value in self.count.items():
            if min_count is not None:
                if value < min_count:
                    continue
            tmp = math.ceil(value ** 0.75)
            total += tmp
            words.append(key)
            probs.append(tmp)
        self.words = words
        self.probs = [prob / total for prob in probs]

    def sample(self, replace=False, min_count=None):
        if not self.if_ready:
            self.ready(min_count=min_count)
            self.if_ready = True
        return np.random.choice(self.words, replace=replace, p=self.probs)
