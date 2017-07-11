import os
import pickle
import fasttext
import numpy as np
from keras.utils import to_categorical
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences


class Dataset(object):
	def __init__(self, train_set=None, test_set=None, word_embeddings_path=None, vocab_size=None, max_seq_length=None, embedding_size=None):
		# file for train set
		self.train_set = train_set
		# file for test set
		self.test_set = test_set
		# file for word embeddings
		self.word_embeddings_path = word_embeddings_path
		# max number of words in the embedding matrix
		self.vocab_size = vocab_size
		# max seq length after padding
		self.max_seq_length = max_seq_length
		# size of word vectors
		self.embedding_size = embedding_size
		# map labels to ids
		self.labels_to_ids = {}
		# load word embeddings
		assert self.word_embeddings_path is not None, "Provide word embeddings path"
		print('Loading word embeddings...')
		if "glove" in word_embeddings_path:
			file_name = "word_embeddings_" + str(embedding_size) + ".p"
			file_name = os.path.join(self.word_embeddings_path, file_name)
			self.embeddings_index = pickle.load(open(file_name, "rb"))
		elif "fasttext" in word_embeddings_path:
			self.embeddings_index = fasttext.load_model(self.word_embeddings_path +  "wiki.en.bin")
		else:
			raise("Word embeddings type not supported")

	def __load_data(self, file_name, separator="\t\t\t"):
		with open(file_name, "r") as f:
			# keep sentences
			texts = []
			# keep labels
			labels = []
			for line in f:
				line = line.strip()
				# extract text
				text = line.split(separator)[0]
				# extract label
				label = line.split(separator)[1]
				if label not in self.labels_to_ids.keys():
					self.labels_to_ids[label] = len(self.labels_to_ids)
				label = self.labels_to_ids[label]
				texts.append(text)
				labels.append(label)
		return texts, labels

	def __prepare_embedding_matrix(self, id_to_word):
		print('Preparing embedding matrix...')
		num_words = min(self.vocab_size, len(id_to_word)+1)
		# initialize embedding matrix
		embedding_matrix = np.zeros((num_words, self.embedding_size))
		for key, value in id_to_word.items():
			word = value
			if key >= self.vocab_size:
				continue
			# take word embedding of token
			if "fasttext" in self.word_embeddings_path:
				embedding_vector = self.embeddings_index[word]
				# normalize
				# embedding_vector /= np.linalg.norm(embedding_vector)
				embedding_matrix[key] = embedding_vector
			if "glove" in self.word_embeddings_path:
				if word in self.embeddings_index.keys():
					embedding_vector = self.embeddings_index[word]
					# normalize
					# embedding_vector /= np.linalg.norm(embedding_vector)
					embedding_matrix[key] = embedding_vector
		return embedding_matrix

	def prepare_data(self, sep="\t\t\t"):
		# loading training data in primitive format (string)
		print("Loading training data...")
		train_texts, train_labels = self.__load_data(self.train_set, sep)
		print('Found %s datapoints for training data.' % len(train_texts))
		# tokenizing and indexing data
		print("Tokenizing and indexing training data...")
		# by default the tokenizer filters punctuation, etc.
		# we change this behavior using the filters parameter
		print(self.vocab_size)
		tokenizer = Tokenizer(filters="", num_words=self.vocab_size)
		tokenizer.fit_on_texts(train_texts)
		# take word mapper
		word_to_id = tokenizer.word_index
		id_to_word = {value: key for key, value in word_to_id.items()}
		print('Found %s unique tokens.' % len(word_to_id))
		# convert list of strings to list of integers
		train_texts_to_id = tokenizer.texts_to_sequences(train_texts)
		# padding sequences
		x_train = pad_sequences(train_texts_to_id, maxlen=self.max_seq_length, padding='post', truncating='post')
		# converting labels to categorical
		y_train = to_categorical(np.asarray(train_labels))

		# loading test data in primitive format (string)
		test_texts, test_labels = self.__load_data(self.test_set, sep)
		print('Found %s datapoints for testing data.' % len(test_texts))
		test_texts_to_id = tokenizer.texts_to_sequences(test_texts)
		x_test = pad_sequences(test_texts_to_id, maxlen=self.max_seq_length, padding='post', truncating='post')

		# rename test labels
		y_test = test_labels

		# preparing word embeddings
		embedding_matrix = self.__prepare_embedding_matrix(id_to_word)
		return x_train, y_train, x_test, y_test, embedding_matrix, len(embedding_matrix), self.labels_to_ids
