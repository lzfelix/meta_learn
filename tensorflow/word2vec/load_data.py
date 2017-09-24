import os
import zipfile
from six.moves import urllib
import random
import collections

import numpy as np
import tensorflow as tf

# Step 1: Download the data.
url = 'http://mattmahoney.net/dc/'

def maybe_download(filename, expected_bytes):
    """Download a file if not present, and make sure it's the right size."""

    if not os.path.exists(filename):
        filename, _ = urllib.request.urlretrieve(url + filename, filename)

    statinfo = os.stat(filename)
    if statinfo.st_size == expected_bytes:
        print('Found and verified', filename)
    else:
        print(statinfo.st_size)
        raise Exception('Failed to verify ' + filename + '. Can you get to it with a browser?')

    return filename

# Read the data into a list of strings.
def read_data(filename):
    """Extract the first file enclosed in a zip file as a list of words."""
    with zipfile.ZipFile(filename) as f:
        data = tf.compat.as_str(f.read(f.namelist()[0])).split()
    return data

def build_dataset(words, n_words):
    """Process raw inputs into a dataset."""

    # compute the word frequency on the corpus
    count = [['UNK', -1]]
    count.extend(collections.Counter(words).most_common(n_words - 1))
    dictionary = dict()

    # assign ID for each word
    for word, _ in count:
        dictionary[word] = len(dictionary)

    # keep track of the K most frequent words and pool the rest on UNK
    data = list()
    unk_count = 0
    for word in words:
        if word in dictionary:
            index = dictionary[word]
        else:
            index = 0  # dictionary['UNK']
            unk_count += 1
        data.append(index)
    count[0][1] = unk_count

    # reversed dictionary maps word ids to word strings.
    reversed_dictionary = dict(zip(dictionary.values(), dictionary.keys()))

    # data -> corpus as word ids.
    # count -> word_frequencies table. word -> frequency
    # dictionary: word -> word_id
    # reverse_dictionary: word_id -> word
    return data, count, dictionary, reversed_dictionary

# Step 3: Function to generate a training batch for the skip-gram model.

data_index = 0
def generate_batch(data, batch_size, num_skips, skip_window):
    global data_index

    assert batch_size % num_skips == 0
    assert num_skips <= 2 * skip_window

    batch  = np.ndarray(shape=(batch_size), dtype=np.int32)
    labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)

    span = 2 * skip_window + 1  # [ skip_window target skip_window ]
    buffer = collections.deque(maxlen=span)

    if data_index + span > len(data):
        data_index = 0

    buffer.extend(data[data_index:data_index + span])
    data_index += span

    for i in range(batch_size // num_skips):
        target = skip_window  # target label at the center of the buffer
        targets_to_avoid = [skip_window]

        for j in range(num_skips):
            # don't repeat samplings
            while target in targets_to_avoid:
                target = random.randint(0, span - 1)

            targets_to_avoid.append(target)
            batch[i * num_skips + j] = buffer[skip_window]
            labels[i * num_skips + j, 0] = buffer[target]

        if data_index == len(data):
            buffer[:] = data[:span]
            data_index = span
        else:
            buffer.append(data[data_index])
            data_index += 1

    # Backtrack a little bit to avoid skipping words in the end of a batch
    data_index = (data_index + len(data) - span) % len(data)
    return batch, labels