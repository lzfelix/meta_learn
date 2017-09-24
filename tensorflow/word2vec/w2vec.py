import math
import os
import zipfile
import random

import tensorflow as tf
import numpy as np

from load_data import  *
from t_sne import plot_tsne

# Loading raw data (1 wikipedia entry per line
filename = maybe_download('text8.zip', 31344016)
vocabulary = read_data(filename)
print('Data size', len(vocabulary))

# Step 2: Build the dictionary and replace rare words with UNK token.
vocabulary_size = 50000
data, count, dictionary, reverse_dictionary = build_dataset(vocabulary,
                                                            vocabulary_size)
del vocabulary # Hint to reduce memory.

print('Most common words (+UNK)', count[:5])
print('Sample data', data[:10], [reverse_dictionary[i] for i in data[:10]])

# batch, labels = generate_batch(data, batch_size=8, num_skips=2, skip_window=1)
# for i in range(8):
#     print(batch[i], reverse_dictionary[batch[i]], '->', labels[i, 0], reverse_dictionary[labels[i, 0]])


batch_size     = 128
embedding_size = 128    # Dimension of the embedding vector.
skip_window    = 1      # How many words to consider left and right.
num_skips      = 2      # How many times to reuse an input to generate a label.

# We pick a random validation set to sample nearest neighbors. Here we limit the
# validation samples to the words that have a low numeric ID, which by
# construction are also the most frequent.
valid_size = 16         # Random set of words to evaluate similarity on.
valid_window = 100      # Only pick dev samples in the head of the distribution.
num_sampled = 64        # Number of negative examples to sample.
valid_examples = np.random.choice(valid_window, valid_size, replace=False)

graph = tf.Graph()
with graph.as_default():

    # Input data slots
    train_inputs = tf.placeholder(tf.int32, shape=[batch_size])
    train_labels = tf.placeholder(tf.int32, shape=[batch_size, 1])
    valid_dataset = tf.constant(valid_examples, dtype=tf.int32)

    # W_{embed} = R^{|V|xd}
    embeddings = tf.Variable(tf.random_uniform([vocabulary_size,
                                                embedding_size], -1.0, 1.0),
                             name="embedding_layer")

    # W_{weights} = R^{|V|xd} -> Weight-sample multiplication
    nce_weights = tf.Variable(tf.truncated_normal([vocabulary_size,
                                                   embedding_size],
                                                  stddev=1.0 / math.sqrt(embedding_size)),
                              name="scoring_layer")

    nce_biases = tf.Variable(tf.zeros([vocabulary_size]), name="scoring_biases")

    # this is the embedding operation. Since this is skip-gram, retrieve 1 vector per input word, represented by wordID
    embed = tf.nn.embedding_lookup(embeddings, train_inputs)

    # Compute the average Noise Contrastive Estimation loss for the batch.
    # tf.nce_loss automatically draws a new sample of the negative labels each time we evaluate the loss.
    loss = tf.reduce_mean(
        tf.nn.nce_loss(weights=nce_weights,
                       biases=nce_biases,
                       labels=train_labels,
                       inputs=embed,
                       num_sampled=num_sampled,
                       num_classes=vocabulary_size))

    # Construct the SGD optimizer using a learning rate of 1.0.
    # The word vectors learning ends here.
    optimizer = tf.train.GradientDescentOptimizer(1.0).minimize(loss)

    # This part is used to retrieve the top-related words.
    # Normalization is used to make vectors multiplication equivalent to the calculation of cosine between them
    norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))
    normalized_embeddings = embeddings / norm

    # This part is only used on validation time. It draws the validation dataset word embeddings from the matrix.
    valid_embeddings = tf.nn.embedding_lookup(normalized_embeddings, valid_dataset)
    similarity = tf.matmul(valid_embeddings, normalized_embeddings, transpose_b=True)

    # The 0-th step of the graph, variable initialization.
    init = tf.global_variables_initializer()

# Step 5: Begin training.
num_steps = 100001

with tf.Session(graph=graph) as session:
    # initializing variables
    init.run()
    print('Initialized')

    average_loss = 0
    for step in range(num_steps):
        # fetching calculation data
        batch_inputs, batch_labels = generate_batch(data, batch_size, num_skips, skip_window)

        feed_dict = {train_inputs: batch_inputs,
                     train_labels: batch_labels}

        # We perform one update step by evaluating the optimizer op (including it
        # in the list of returned values for session.run()
        _, loss_val = session.run([optimizer, loss], feed_dict=feed_dict)
        average_loss += loss_val

        if step % 2000 == 0:
            # The average loss is an estimate of the loss over the last 2000 batches.
            if step > 0:
                average_loss /= 2000

            print('Average loss at step ', step, ': ', average_loss)
            average_loss = 0

        # Note that this is expensive (~20% slowdown if computed every 500 steps)
        if step % 10000 == 0:
            sim = similarity.eval()
            for i in range(valid_size):
                valid_word = reverse_dictionary[valid_examples[i]]
                top_k = 8  # number of nearest neighbors
                nearest = (-sim[i, :]).argsort()[1:top_k + 1]
                log_str = 'Nearest to %s:' % valid_word
                for k in range(top_k):
                    close_word = reverse_dictionary[nearest[k]]
                    log_str = '%s %s,' % (log_str, close_word)
                print(log_str)

    final_embeddings = normalized_embeddings.eval()

# Step 6: plot first 500 words using T-SNE
plot_tsne(reverse_dictionary, final_embeddings)
