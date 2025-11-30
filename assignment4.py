import math
import os

class BagOfWordsModel:
	def __init__(self, directory):
		# directory is the full path to a directory containing trials through state space
		docWords = []
		wordFreq = {}
		docCount = 0
		self.idf = {}

		fileNames = os.listdir(directory)
		files = [f for f in fileNames if f.endswith(".txt") and os.path.isfile(os.path.join(directory, f))]
		for file in files:
			docCount += 1
			filePath = os.path.join(directory, file)
			with open(filePath, newline='', mode='r') as f:
				content = f.read()
				uniqueList = list(set(content.split()))
				for word in uniqueList:
					wordFreq[word] = wordFreq.get(word, 0) + 1
				docWords += content.split()

		wordFreq = dict(sorted(wordFreq.items()))
		for word in wordFreq:
			self.idf[word] = math.log(docCount / wordFreq.get(word), 2)

		print(self.idf)
		# Return nothing


	def tf_idf(self, document_filepath):
		# document_filepath is the full file path to a test document
		pass
		# Return the term frequency-inverse document frequency vector for the document
		# return tf_idf_vector


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