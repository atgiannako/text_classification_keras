import os
from parameters import parameters

import numpy as np
from keras.models import Model
from keras.layers.core import Flatten
from keras.layers import GRU, Bidirectional
from keras.layers import Input, Embedding, Dense
from keras.layers import Conv1D, MaxPooling1D, concatenate 


class NeuralNets(object):
	def __init__(self, embedding_matrix=None, max_seq_length=None, max_num_words=None, embedding_size=None, batch_size=None, epochs=None, labels_to_ids=None):
		# embeddings matrix
		self.embedding_matrix = embedding_matrix
		# max seq length after padding
		self.max_seq_length = max_seq_length
		# max number of words
		self.max_num_words = max_num_words
		# size of word vectors
		self.embedding_size = embedding_size
		# define batch size
		self.batch_size = batch_size
		# define epochs for training
		self.epochs = epochs
		# labels to ids
		self.labels_to_ids = labels_to_ids
		# id to labels
		self.id_to_label = {value: key for key, value in self.labels_to_ids.items()}
		# number of classes
		self.num_of_classes = len(labels_to_ids)

	def __parse_predictions(self, output_file, predictions, y_test):
		y_hat = []
		for prediction in predictions:
		    y_hat.append(np.argmax(prediction))
		predictions = y_hat
		# materialize predictions
		with open(output_file, "w") as f:
			for i in range(len(predictions)):
				pred_label = self.id_to_label[predictions[i]]
				true_label = self.id_to_label[y_test[i]]
				print(pred_label)
				f.write(str(i+1) + " " + true_label + " " + pred_label + "\n")

	def cnn(self, x_train, y_train, x_test, y_test, filters, kernel_sizes):
		# give input
		main_input = Input(shape=(self.max_seq_length, ), dtype='int32', name='main_input')
		# embedding layer
		embeddings = Embedding(self.max_num_words, self.embedding_size, input_length=self.max_seq_length, weights=[self.embedding_matrix], trainable=False)(main_input)
		if len(kernel_sizes) == 1:
			print("Building CNN model with single kernel size...")
			kernel_size = kernel_sizes[0]
			# convolution layer
			conv1d = Conv1D(filters, kernel_size, padding='valid', activation='relu')(embeddings)
			# max pooling
			max_pool = (MaxPooling1D(pool_size=self.max_seq_length - kernel_size + 1)(conv1d))
			# flatten output
			flatten = Flatten()(max_pool)
		else:
			print("Building CNN model with multiple kernel sizes...")
			convs = []
			# perform convolution for each kernel size
			for kernel_size in kernel_sizes:
			    # convolution layer
			    conv1d = Conv1D(filters, kernel_size, padding='valid', activation='relu')(embeddings)
			    # max pooling
			    max_pool = (MaxPooling1D(pool_size=self.max_seq_length - kernel_size + 1)(conv1d))
			    convs.append(max_pool)
			# concatenate filter outputs
			merged_convs = concatenate(convs)
			# flatten output
			flatten = Flatten()(merged_convs)
		# output layer with softmax activation
		predictions = Dense(self.num_of_classes, activation='softmax')(flatten)
		print("Training CNN model...")
		model = Model(inputs=main_input, outputs=predictions)
		model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
		# train model
		model.fit(x_train, y_train, batch_size=self.batch_size, epochs=self.epochs, shuffle=False)
		print("Predicting...")
		# make predictions
		predictions = model.predict(x_test, batch_size=self.batch_size)
		# parse predictions
		output_file = os.path.join(parameters["output_directory"], "predictions.txt")
		self.__parse_predictions(output_file, predictions, y_test)
		# evaluate performance
		print("\n\nPerformance Evaluation:")
		command = "perl conll.pl -r < " + output_file
		os.system(command)

	def gru(self, x_train, y_train, x_test, y_test, num_of_units, bidirectional=True):
		# give input
		main_input = Input(shape=(self.max_seq_length, ), dtype='int32', name='main_input')
		# embedding layer
		embeddings = Embedding(self.max_num_words, self.embedding_size, input_length=self.max_seq_length, weights=[self.embedding_matrix], trainable=False)(main_input)
		if bidirectional:
			print("Building B-GRU model...")
			# build bidirectional GRU model
			gru_out = Bidirectional(GRU(num_of_units))(embeddings)
		else:
			print("Building GRU model...")
			# build simple GRU model
			gru_out = GRU(num_of_units)(embeddings)
		# output layer with softmax activation
		predictions = Dense(self.num_of_classes, activation='softmax')(gru_out)
		print("Training GRU model...")
		model = Model(inputs=main_input, outputs=predictions)
		model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
		# train model
		model.fit(x_train, y_train, batch_size=self.batch_size, epochs=self.epochs, shuffle=False)
		print("Predicting...")
		# make predictions
		predictions = model.predict(x_test, batch_size=self.batch_size)
		# parse predictions
		output_file = "predictions.txt"
		self.__parse_predictions(output_file, predictions, y_test)
		# evaluate performance
		print("\n\nPerformance Evaluation:")
		command = "perl conll.pl -r < predictions.txt"
		os.system(command)

	def cnn_gru(self, x_train, y_train, x_test, y_test, filters, kernel_sizes, num_of_units, bidirectional=True):
		# give input
		main_input = Input(shape=(self.max_seq_length, ), dtype='int32', name='main_input')
		# embedding layer
		embeddings = Embedding(self.max_num_words, self.embedding_size, input_length=self.max_seq_length, weights=[self.embedding_matrix], trainable=False)(main_input)
		if len(kernel_sizes) == 1:
			print("Convolution layer with single kernel size...")
			kernel_size = kernel_sizes[0]
			# convolution layer
			conv1d = Conv1D(filters, kernel_size, padding='valid', activation='relu')(embeddings)
			# max pooling
			conv_output = (MaxPooling1D(pool_size=self.max_seq_length - kernel_size + 1)(conv1d))
		else:
			print("Convolution layer with multiple kernel sizes...")
			convs = []
			# perform convolution for each kernel size
			for kernel_size in kernel_sizes:
			    # convolution layer
			    conv1d = Conv1D(filters, kernel_size, padding='valid', activation='relu')(embeddings)
			    # max pooling
			    max_pool = (MaxPooling1D(pool_size=self.max_seq_length - kernel_size + 1)(conv1d))
			    convs.append(max_pool)
			# concatenate filter outputs
			conv_output = concatenate(convs)
		if bidirectional:
			print("B-GRU layer...")
			# build bidirectional GRU model
			gru_out = Bidirectional(GRU(num_of_units))(conv_output)
		else:
			print("GRU layer...")
			# build simple GRU model
			gru_out = GRU(num_of_units)(conv_output)
		# output layer with softmax activation
		predictions = Dense(self.num_of_classes, activation='softmax')(gru_out)
		print("Training GRU model...")
		model = Model(inputs=main_input, outputs=predictions)
		model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
		# train model
		model.fit(x_train, y_train, batch_size=self.batch_size, epochs=self.epochs, shuffle=False)
		print("Predicting...")
		# make predictions
		predictions = model.predict(x_test, batch_size=self.batch_size)
		# parse predictions
		output_file = "predictions.txt"
		self.__parse_predictions(output_file, predictions, y_test)
		# evaluate performance
		print("\n\nPerformance Evaluation:")
		command = "perl conll.pl -r < predictions.txt"
		os.system(command)