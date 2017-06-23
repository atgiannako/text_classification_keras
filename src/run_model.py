import sys
from Dataset import *
from NeuralNets import NeuralNets
from parameters import parameters


# Parameters
train_set = parameters["train_file"]
test_set = parameters["test_file"]
max_seq_length = parameters["MAX_SEQ_LENGTH"]
max_num_words = parameters["max_num_words"]
embedding_size = parameters["embedding_size"]
word_embeddings_path = parameters["word_embeddings_path"]
batch_size = parameters["batch_size"]
epochs = parameters["epochs"]
kernel_sizes = parameters["kernel_sizes"]
filters = parameters["filters"]
num_of_units = parameters["num_of_units"]

data = Dataset(train_set, test_set, word_embeddings_path, max_num_words, max_seq_length, embedding_size)
x_train, y_train, x_test, y_test, embedding_matrix, vocab_size, labels_to_ids = data.prepare_data()

# the vocab size may be different if the initial max_num_words is greater than the dataset vocabulary
max_num_words = vocab_size

nn = NeuralNets(embedding_matrix, max_seq_length, max_num_words, embedding_size, batch_size, epochs, labels_to_ids)

# run CNN model
nn.cnn(x_train, y_train, x_test, y_test, filters, kernel_sizes)

# run GRU/LSTM model
nn.gru(x_train, y_train, x_test, y_test, num_of_units, bidirectional=True)
