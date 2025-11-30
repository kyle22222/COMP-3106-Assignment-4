import os

class BagOfWordsModel:
	def __init__(self, directory):
		# directory is the full path to a directory containing trials through state space
		docWords = []

		fileNames = os.listdir(directory)
		files = [f for f in fileNames if f.endswith(".txt") and os.path.isfile(os.path.join(directory, f))]
		for file in files:
			filePath = os.path.join(directory, file)
			with open(filePath, newline='', mode='r') as f:
				content = f.read()
				docWords += content.split()

		print(docWords)
		# Return nothing


  def tf_idf(self, document_filepath):
    with open(document_filepath, 'r') as f:
        content = f.read()

    tokens = content.split()
    total_words = len(tokens)

    if  total_words == 0:
        return [0.0] * len(self.vocabulary)

    doc_word_counts = {}
    for token in tokens:
        doc_word_counts[token] = doc_word_counts.get(token, 0) + 1

    tf_idf_vector = []

    for i, term in enumerate(self.vocabulary):
        term_count = doc_word_counts.get(term, 0)
        tf = term_count / total_words

        idf = self.idf[i]

        tf_idf_vector.append(tf * idf)

    return tf_idf_vector


	def predict(self, document_filepath, business_weights, entertainment_weights, politics_weights):
		# document_filepath is the full file path to a test document
		# business_weights is a list of weights for the business artificial neuron
		# entertainment_weights is a list of weights for the entertainment artificial neuron
		# politics_weights is a list of weights for the politics artificial neuron
		pass
		# Return the predicted label from the neural network model
		# Return the score from each neuron
		# return predicted_label, scores

if __name__ == '__main__':
	bagOfWords0 = BagOfWordsModel("./Examples/Example0/training_documents")
	bagOfWords1 = BagOfWordsModel("./Examples/Example1/training_documents")
	bagOfWords2 = BagOfWordsModel("./Examples/Example2/training_documents")