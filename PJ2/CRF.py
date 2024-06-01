import sklearn_crfsuite

class CRF:
    def __init__(self):
        self.model = sklearn_crfsuite.CRF(
            algorithm='lbfgs', # Gradient descent using the L-BFGS method
            c1=0.1,
            c2=0.1,
            max_iterations=100,
            all_possible_transitions=True
        )
    
    def fit(self, X, Y):
        X_ = []
        Y_ = []
        for i in range(len(X)):
            tmpX = []
            for j in range(len(X[i])):
                tmpX.append(word2features(X[i], j))
            X_.append(tmpX)
        for i in range(len(Y)):
            tmpY = []
            for j in range(len(Y[i])):
                tmpY.append(Y[i][j])
            Y_.append(tmpY)
        return self.model.fit(X_, Y_)
    
    def predict(self, X):
        X_ = []
        for i in range(len(X)):
            tmpX = []
            for j in range(len(X[i])):
                tmpX.append(word2features(X[i], j))
            X_.append(tmpX)
        return self.model.predict(X_)
    
    def score(self, X, y):
        return self.model.score(X, y)
    
def word2features(sent, i):

    word = sent[i]
    features = {
        'bias': 1.0,
        'word': word,
        'word.isdigit()': word.isdigit(),
    }
    if i > 0:
        word1 = sent[i-1]
        words = word1 + word
        features.update({
            '-1:word': word1,
            '-1:words': words,
            '-1:word.isdigit()': word1.isdigit(),
        })
    else:
        features['BOS'] = True
    if i > 1:
        word2 = sent[i-2]
        word1 = sent[i-1]
        words = word1 + word2 + word
        features.update({
            '-2:word': word2,
            '-2:words': words,
            '-3:word.isdigit()': word2.isdigit(),
        })
    if i > 2:
        word3 = sent[i - 3]
        word2 = sent[i - 2]
        word1 = sent[i - 1]
        words = word1 + word2 + word3 + word
        features.update({
            '-3:word': word3,
            '-3:words': words,
            '-3:word.isdigit()': word3.isdigit(),
        })
    if i > 3:
        word4 = sent[i - 4]
        word3 = sent[i - 3]
        word2 = sent[i - 2]
        word1 = sent[i - 1]
        words = word1 + word2 + word3 + word4 + word
        features.update({
            '-4:word': word4,
            '-4:words': words,
            '-4:word.isdigit()': word4.isdigit(),
        })
    if i < len(sent)-1:
        word1 = sent[i+1]
        words = word1 + word
        features.update({
            '+1:word': word1,
            '+1:words': words,
            '+1:word.isdigit()': word1.isdigit(),
        })
    else:
        features['EOS'] = True
    if i < len(sent)-2:
        word2 = sent[i + 2]
        word1 = sent[i + 1]
        words = word + word1 + word2
        features.update({
            '+2:word': word2,
            '+2:words': words,
            '+2:word.isdigit()': word2.isdigit(),
        })
    if i < len(sent)-3:
        word3 = sent[i + 3]
        word2 = sent[i + 2]
        word1 = sent[i + 1]
        words = word + word1 + word2 + word3
        features.update({
            '+3:word': word3,
            '+3:words': words,
            '+3:word.isdigit()': word3.isdigit(),
        })
    features['word.isupper'] = word.isupper()
    features['word.islower'] = word.islower()
    features['word.position'] = i / len(sent)
    return features