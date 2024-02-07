import os
import tarfile

import numpy as np
import gdown
import tensorflow_datasets as tfds
import tensorflow as tf
# Ensure TF does not see GPU and grab all GPU memory.
tf.config.set_visible_devices([], device_type='GPU')

options = tf.data.Options()
options.deterministic = True


def datasets_to_dataloaders(train_dataset, val_dataset, test_dataset, batch_size, drop_remainder=True, transform=None):
    # Shuffle train dataset
    train_dataset = train_dataset.shuffle(10_000, reshuffle_each_iteration=True)

    # Batch
    train_dataset = train_dataset.batch(batch_size, drop_remainder=drop_remainder)
    val_dataset = val_dataset.batch(batch_size, drop_remainder=drop_remainder)
    test_dataset = test_dataset.batch(batch_size, drop_remainder=drop_remainder)

    # Transform
    if transform is not None:
        train_dataset = train_dataset.map(transform, num_parallel_calls=tf.data.AUTOTUNE)
        val_dataset = val_dataset.map(transform, num_parallel_calls=tf.data.AUTOTUNE)
        test_dataset = test_dataset.map(transform, num_parallel_calls=tf.data.AUTOTUNE)

    # Prefetch
    train_dataset = train_dataset.prefetch(tf.data.AUTOTUNE)
    val_dataset = val_dataset.prefetch(tf.data.AUTOTUNE)
    test_dataset = test_dataset.prefetch(tf.data.AUTOTUNE)

    # Convert to NumPy for JAX
    return tfds.as_numpy(train_dataset), tfds.as_numpy(val_dataset), tfds.as_numpy(test_dataset)

def get_imdb_dataloaders(data_dir: str = '~/data', batch_size: int = 1, drop_remainder: bool = True,
                         max_vocab_size: int = 20_000, max_seq_len: int = 512):
    """
    Returns dataloaders for the IMDB sentiment analysis dataset (natural language processing, binary classification),
    as well as the vocabulary and tokenizer.

    Information about the dataset: https://www.tensorflow.org/datasets/catalog/imdb_reviews
    """
    import tensorflow_text as tf_text
    from tensorflow_text.tools.wordpiece_vocab.bert_vocab_from_dataset import bert_vocab_from_dataset

    data_dir = os.path.expanduser(data_dir)

    # Load datasets
    train_dataset, val_dataset, test_dataset = tfds.load(name='imdb_reviews',
                                                         split=['train[:90%]', 'train[90%:]', 'test'], as_supervised=True, data_dir=data_dir, shuffle_files=True)
    train_dataset, val_dataset, test_dataset = train_dataset.with_options(options), val_dataset.with_options(options), test_dataset.with_options(options)
    print("Cardinalities (train, val, test):", train_dataset.cardinality().numpy(), val_dataset.cardinality().numpy(), test_dataset.cardinality().numpy())

    # Build vocabulary and tokenizer
    bert_tokenizer_params = dict(lower_case=True)
    vocab = bert_vocab_from_dataset(
        train_dataset.batch(10_000).prefetch(tf.data.AUTOTUNE).map(lambda x, _: x),
        vocab_size=max_vocab_size,
        reserved_tokens=["[PAD]", "[UNK]", "[START]", "[END]"],
        bert_tokenizer_params=bert_tokenizer_params
    )
    vocab_lookup_table = tf.lookup.StaticVocabularyTable(
        num_oov_buckets=1,
        initializer=tf.lookup.KeyValueTensorInitializer(keys=vocab,
                                                        values=tf.range(len(vocab), dtype=tf.int64))  # setting tf.int32 here causes an error
    )
    tokenizer = tf_text.BertTokenizer(vocab_lookup_table, **bert_tokenizer_params)

    def preprocess(text, label):
        # Tokenize
        tokens = tokenizer.tokenize(text).merge_dims(-2, -1)
        # Cast to int32 for compatibility with JAX (note that the vocabulary size is small)
        tokens = tf.cast(tokens, tf.int32)
        # Pad (all sequences to the same length so that JAX jit compiles the model only once)
        padded_inputs, _ = tf_text.pad_model_inputs(tokens, max_seq_length=max_seq_len)
        return padded_inputs, label

    return datasets_to_dataloaders(train_dataset, val_dataset, test_dataset, batch_size,
                                   drop_remainder=drop_remainder, transform=preprocess), vocab, tokenizer
