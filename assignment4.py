import math, os

class BagOfWordsModel:
    def __init__(self, directory):
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

    def tf_idf(self, documentFilepath):
        with open(documentFilepath, 'r') as f:
            content = f.read()

        tokens = content.split()
        totalWords = len(tokens)

        if totalWords == 0:
            return [0.0] * len(self.idf)

        docWordCounts = {}
        for token in tokens:
            docWordCounts[token] = docWordCounts.get(token, 0) + 1

        tfIdfVector = []

        for term in self.idf:
            termCount = docWordCounts.get(term, 0)
            tf = termCount / totalWords
            idf = self.idf[term]
            tfIdfVector.append(tf * idf)

        return tfIdfVector

    @staticmethod
    def _dot(weights, x):
        dotSum = 0
        for w, tf in zip(weights, x):
            dotSum += w * tf
        return dotSum

    @staticmethod
    def _softmax(scores):
        maxScore = max(scores)
        exps = [math.exp(s - maxScore) for s in scores]
        total = sum(exps)
        return [e / total for e in exps]

    def predict(self, documentFilepath, businessWeights, entertainmentWeights, politicsWeights):
        x = self.tf_idf(documentFilepath)

        z1 = self._dot(businessWeights, x)
        z2 = self._dot(entertainmentWeights, x)
        z3 = self._dot(politicsWeights, x)

        scores = self._softmax([z1, z2, z3])
        catStrings = ["business", "entertainment", "politics"]
        predictedLabel = catStrings[max(range(3), key = lambda k: scores[k])]
        return predictedLabel, scores

def readNumbers(filepath):
    with open(filepath, 'r') as f:
        content = f.read()

    return [float(num.strip()) for num in content.split(',')]

if __name__ == '__main__':
    for ex in range(3):
        folder = "./Examples/Example" + str(ex) + "/"
        cats = [readNumbers(folder + cat + "_weights.txt") for cat in ["business", "entertainment", "politics"]]
        bagOfWords = BagOfWordsModel(folder + "training_documents")
        prediction = bagOfWords.predict(folder + "test_document.txt", cats[0], cats[1], cats[2])
        print(prediction)