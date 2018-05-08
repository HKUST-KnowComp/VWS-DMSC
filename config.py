import os
import tensorflow as tf

from main import train

flags = tf.flags
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

flags.DEFINE_integer("aspect", 0, "aspect index")
flags.DEFINE_string("mode", "train", "train/debug/test")
flags.DEFINE_string(
    "aspect_seeds", "data/tripadvisor/aspect.words", "path to aspect seeds")
flags.DEFINE_string("train", "data/tripadvisor/train", "path to train data")
flags.DEFINE_string("dev", "data/tripadvisor/dev", "path to dev data")
flags.DEFINE_string("test", "data/tripadvisor/test", "path to test data")
flags.DEFINE_string("emb", "data/tripadvisor/emb",
                    "path to pre-trained embedding")
flags.DEFINE_string("asp_emb", "data/tripadvisor/ret_emb",
                    "path to pre-trained aspect embedding")


flags.DEFINE_integer("batch_size", 8, "mini-batch size")
flags.DEFINE_integer("test_batch_size", 32, "mini-batch size for test")
flags.DEFINE_float("dropout_rate", 0.3, "dropout rate")
flags.DEFINE_integer("hop_sent", 2, "hop for sentence level")
flags.DEFINE_integer("hop_word", 2, "hop for word level")
flags.DEFINE_float("learning_rate", 0.001, "learning rate")
flags.DEFINE_float("en_l2_reg", 0.001, "l2 reg for encoder")
flags.DEFINE_float("de_l2_reg", 0.001, "l2 reg for decoder")
flags.DEFINE_float("alpha", 0.1, "")
flags.DEFINE_integer("emb_dim", 200, "dimension of embedding matrix")
flags.DEFINE_integer("hidden", 200, "hidden dimension")

flags.DEFINE_integer("cache_size", 100, "size of dataset buffer")

flags.DEFINE_integer("eval_period", 100, "evaluate on dev every period")
flags.DEFINE_boolean("load_pretrain", True, "load overall as pretrain")
flags.DEFINE_integer("max_epochs", 5, "maximum number of epochs")
flags.DEFINE_integer("num_batches", 200, "number of batches in evaluation")

flags.DEFINE_integer("score_scale", 5, "score scale")
flags.DEFINE_integer("num_senti", 5, "number of sentiment word in sampling")
flags.DEFINE_integer("neg_num", 25, "number of negative sampling")
flags.DEFINE_integer("min_count", 3, "min count in batches creation")


def main(_):
    config = flags.FLAGS
    train(config)


if __name__ == "__main__":
    tf.app.run()
