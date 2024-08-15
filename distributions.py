import numpy as np

class GaussianDistribution:
    def __init__(self, num_classes):
        self.num_classes=num_classes

    def fit(self, params):
        self.params=params

    def predict(self, X, priors):
        predictions=[]
        for x in X:
            posterior_probs = {}
            for c in priors:
                likelihood = 1.0
                x_dim = np.hsplit(x, 10)
                likelihood *= (1.0 / ((self.params[c][2])**0.5 * np.sqrt(2 * np.pi))) * np.exp(-((x_dim[0] - self.params[c][0]) ** 2) / (2 * self.params[c][2]))
                likelihood *= (1.0 / ((self.params[c][3])**0.5 * np.sqrt(2 * np.pi))) * np.exp(-((x_dim[1] - self.params[c][1]) ** 2) / (2 * self.params[c][3]))
                posterior_prob = np.log(priors[c]) + np.sum(u7np.log(likelihood))
                posterior_probs[c]=posterior_prob
            predicted_class = max(posterior_probs, key= lambda x: posterior_probs[x])
            predictions.append(predicted_class)
        return predictions
    
    def calculateParams(self, X, y):
        gaussian={}
        for c in range(self.num_classes):
            X_c = X[y == c]
            X_dim = np.hsplit(X_c, 10)
            mean_X1 = np.mean(X_dim[0])
            mean_X2 = np.mean(X_dim[1])
            var_X1 = np.var(X_dim[0])
            var_X2 = np.var(X_dim[1])
            gaussian[c] = [mean_X1, mean_X2, var_X1, var_X2]
        return gaussian

class BernoulliDistribution:
    def __init__(self, num_classes):
        self.num_classes=num_classes

    def fit(self, params):
        self.params=params
    
    def predict(self, X, priors):
        predictions = []
        for x in X:
            posterior_probs = {}
            for c in priors:
                likelihood = 1.0
                x_dim = np.hsplit(x, 10)
                likelihood *= (self.params[c][0]**x_dim[2])*((1-self.params[c][0])**(1-x_dim[2]))
                likelihood *= (self.params[c][1]**x_dim[3])*((1-self.params[c][1])**(1-x_dim[3]))
                posterior_prob = np.log(priors[c]) + np.sum(np.log(likelihood))
                posterior_probs[c]=posterior_prob
            predicted_class = max(posterior_probs, key= lambda x: posterior_probs[x])
            predictions.append(predicted_class)
        return predictions
    
    def calculateParams(self, X, y):
        bernoulli={}
        for c in range(self.num_classes):
            X_c = X[y == c]
            X_dim = np.hsplit(X_c, 10)
            p_X3 = np.mean(X_dim[2])
            p_X4 = np.mean(X_dim[3])
            bernoulli[c] = [p_X3, p_X4]
        return bernoulli

class LaplaceDistribution:
    def __init__(self, num_classes):
        self.num_classes=num_classes

    def fit(self, params):
        self.params = params
    
    def predict(self, X, priors):
        predictions = []
        for x in X:
            posterior_probs = {}
            for c in priors:
                likelihood = 1.0
                x_dim = np.hsplit(x, 10)
                likelihood *= (1.0 / (2 * self.params[c][2])) * np.exp(-np.abs(x_dim[4] - self.params[c][0]) / self.params[c][2])
                likelihood *= (1.0 / (2 * self.params[c][3])) * np.exp(-np.abs(x_dim[5] - self.params[c][1]) / self.params[c][3])
                posterior_prob = np.log(priors[c]) + np.sum(np.log(likelihood))
                posterior_probs[c]=posterior_prob
            predicted_class = max(posterior_probs, key= lambda x: posterior_probs[x])
            predictions.append(predicted_class)
        return predictions
    
    def calculateParams(self, X, y):
        laplace={}
        for c in range(self.num_classes):
            X_c = X[y == c]
            X_dim = np.hsplit(X_c, 10)
            mu_X5 = np.mean(X_dim[4])
            mu_X6 = np.mean(X_dim[5])
            b_X5 = np.mean(np.abs(X_dim[4] - mu_X5))
            b_X6 = np.mean(np.abs(X_dim[5] - mu_X6))
            laplace[c] = [mu_X5, mu_X6, b_X5, b_X6]
        return laplace

class ExponentialDistribution:
    def __init__(self, num_classes):
        self.num_classes=num_classes

    def fit(self, params):
        self.params=params

    def predict(self, X, priors):
        predictions = []
        for x in X:
            posterior_probs = {}
            for c in priors:
                likelihood = 1.0
                x_dim = np.hsplit(x, 10)
                likelihood *= self.params[c][0] * np.exp(-self.params[c][0] * x_dim[6])
                likelihood *= self.params[c][1] * np.exp(-self.params[c][1] * x_dim[7])
                posterior_prob = np.log(priors[c]) + np.sum(np.log(likelihood))
                posterior_probs[c]=posterior_prob
            predicted_class = max(posterior_probs, key= lambda x: posterior_probs[x])
            predictions.append(predicted_class)
        return predictions
    
    def calculateParams(self, X, y):
        exponential={}
        for c in range(self.num_classes):
            X_c = X[y == c]
            X_dim = np.hsplit(X_c, 10)
            lambda_X7 = 1 / np.mean(X_dim[6])
            lambda_X8 = 1 / np.mean(X_dim[7])
            exponential[c] = [lambda_X7, lambda_X8]
        return exponential

class MultinomialDistribution:
    def __init__(self, num_classes):
        self.num_classes=num_classes

    def fit(self, params):
        self.params=params

    def predict(self, X, priors):
        predictions = []
        for x in X:
            posterior_probs = {}
            for c in priors:
                likelihood = 1.0
                x_dim = np.hsplit(x, 10)
                likelihood *= self.params[c][0][int(x_dim[8][0])]
                likelihood *= self.params[c][1][int(x_dim[9][0])]
                posterior_prob = np.log(priors[c]) + np.sum(np.log(likelihood))
                posterior_probs[c]=posterior_prob
            predicted_class = max(posterior_probs, key= lambda x: posterior_probs[x])
            predictions.append(predicted_class)
        return predictions
    
    def calculateParams(self, X, y):
        multinomial={}
        for c in range(self.num_classes):
            X_c = X[y == c]
            X_dim = np.hsplit(X_c, 10)
            numDiffValues = len(np.unique(X_dim[8]))
            p_X9=[]
            for i in range(numDiffValues):
                p_X9.append(np.sum(X_dim[8] == i) / len(X_dim[8]))
            numDiffValues = len(np.unique(X_dim[9]))
            p_X10=[]
            for i in range(numDiffValues):
                p_X10.append(np.sum(X_dim[9] == i) / len(X_dim[9]))
            multinomial[c] = [p_X9, p_X10]
        return multinomial
