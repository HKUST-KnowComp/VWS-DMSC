import os
import tensorflow as tf

from main import train

flags = tf.flags
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

flags.DEFINE_integer("aspect", 0, "aspect to use in unsupervised learning")
flags.DEFINE_integer("num_aspects", 4, "total number of aspects")
flags.DEFINE_list("name_aspects", ["feel", "look", "smell", "taste"], "name of aspects")
flags.DEFINE_string("aspect_seeds", "data/beer/aspect.words", "path to aspect seeds")
flags.DEFINE_string("train", "data/beer/train", "path to train data")
flags.DEFINE_string("dev", "data/beer/dev", "path to dev data")
flags.DEFINE_string("test", "data/beer/test", "path to test data")
flags.DEFINE_string("emb", "data/beer/ret_emb", "path to pre-trained embedding")
flags.DEFINE_string("asp_emb", "data/beer/ret_emb", "path to pre-trained aspect embedding")


flags.DEFINE_integer("batch_size", 8, "mini-batch size")
flags.DEFINE_float("keep_prob", 0.7, "dropout rate")
flags.DEFINE_integer("hop_word", 6, "hop for word level")
flags.DEFINE_integer("hop_sent", 2, "hop for sentence level")
flags.DEFINE_float("learning_rate", 1.0, "learning rate for adadelta")
flags.DEFINE_float("en_l2_reg", 0.00001, "l2 reg for encoder")
flags.DEFINE_float("de_l2_reg", 0.00001, "l2 reg for decoder")
flags.DEFINE_float("alpha", 0.1, "")
flags.DEFINE_integer("emb_dim", 200, "dimension of embedding matrix")
flags.DEFINE_integer("hidden", 200, "hidden dimension")

flags.DEFINE_integer("cache_size", 500, "size of dataset buffer")
flags.DEFINE_string("log_dir", "log/beer/", "directory for saving log")
flags.DEFINE_string("save_dir", "model/beer/", "directory for saving model")

flags.DEFINE_integer("record_period", 100, "record loss every period")
flags.DEFINE_integer("eval_period", 1000, "evaluate on dev every period")
flags.DEFINE_integer("num_epochs", 10, "maximum number of epochs")
flags.DEFINE_integer("num_batches", 200, "number of batches in evaluation")

flags.DEFINE_integer("score_scale", 2, "score scale")
flags.DEFINE_integer("num_senti", 5, "number of sentiment word in sampling")
flags.DEFINE_integer("neg_num", 50, "number of negative sampling")
flags.DEFINE_integer("min_count", 3, "min count in batches creation")
flags.DEFINE_boolean("overall", False, "whether to use overall")
flags.DEFINE_boolean("unsupervised", True, "whether to use unsupervised method")
flags.DEFINE_integer("max_to_keep", 5, "number of models to save")


def main(_):
    config = flags.FLAGS
    train(config)


if __name__ == "__main__":
    tf.app.run()
