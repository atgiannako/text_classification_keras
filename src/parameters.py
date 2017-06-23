parameters = {
	# structure of sentence
	"MAX_SEQ_LENGTH" : , # e.g. 50
	
	# Embedding parameters
	"max_num_words" : , # e.g. 2000
	"embedding_size" : , # e.g. 300
	"word_embeddings_path" : "", # e.g. "../glove/"

	# Convolution parameters
	"kernel_sizes" : , # e.g. [5], [3,5] 
	"filters" : , # e.g. 512

	# GRU / LSTM parameters
	"num_of_units" : , # e.g. 128

	# Training parameters
	"batch_size" : , # e.g. 64
	"epochs" : , # e.g. 10

	# Test and train set
	"train_file" : "", # e.g. ../datasets/toy_train.txt
	"test_file" : "", # e.g. ../datasets/toy_test.txt

	# output directory
	"output_directory": "" # e.g. ../output/
}