import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle as pkl
from distributions import GaussianDistribution
from distributions import BernoulliDistribution
from distributions import LaplaceDistribution
from distributions import ExponentialDistribution
from distributions import MultinomialDistribution

class NaiveBayes:
    def fit(self, X, y):
        self.X=X
        self.y=y
        self.num_classes=len(np.unique(y))

        self.gaussianObject=GaussianDistribution(self.num_classes)
        self.bernoulliObject=BernoulliDistribution(self.num_classes)
        self.laplaceObject=LaplaceDistribution(self.num_classes)
        self.exponentialObject=ExponentialDistribution(self.num_classes)
        self.multinomialObject=MultinomialDistribution(self.num_classes)

        priors,guassian,bernoulli,laplace,exponential,multinomial=self.getParams()
        self.priors=priors
        self.gaussianObject.fit(guassian)
        self.bernoulliObject.fit(bernoulli)
        self.laplaceObject.fit(laplace)
        self.exponentialObject.fit(exponential)
        self.multinomialObject.fit(multinomial)


    def predict(self, X):
        gaussianPredictions = self.gaussianObject.predict(X,self.priors)
        bernoulliPredictions = self.bernoulliObject.predict(X, self.priors)
        laplacePredictions = self.laplaceObject.predict(X, self.priors)
        exponentialPredictions = self.exponentialObject.predict(X, self.priors)
        multinomialPredictions = self.multinomialObject.predict(X, self.priors)
        predictions=[]
        for i in range(X.shape[0]):
            d=[0]*self.num_classes
            d[gaussianPredictions[i]]+=1
            d[bernoulliPredictions[i]]+=1
            d[laplacePredictions[i]]+=1
            d[exponentialPredictions[i]]+=1
            d[multinomialPredictions[i]]+=1
            predictions.append(np.argmax(d))
        return np.array(predictions)

    def getParams(self):
        priors = self.getPriors(self.y)
        guassian = self.gaussianObject.calculateParams(self.X, self.y)
        bernoulli = self.bernoulliObject.calculateParams(self.X, self.y)
        laplace = self.laplaceObject.calculateParams(self.X, self.y)
        exponential = self.exponentialObject.calculateParams(self.X, self.y)
        multinomial = self.multinomialObject.calculateParams(self.X, self.y)
        return (priors, guassian, bernoulli, laplace, exponential, multinomial)        

    def getPriors(self, y):
        priors = {}
        for c in range(self.num_classes):
            priors[c] = np.sum(y == c) / len(y)
        return priors

def save_model(model,filename="model.pkl"):
    """
    You are not required to modify this part of the code.
    """
    file = open("model.pkl","wb")
    pkl.dump(model,file)
    file.close()

def load_model(filename="model.pkl"):
    """
    You are not required to modify this part of the code.
    """
    file = open(filename,"rb")
    model = pkl.load(file)
    file.close()
    return model

def visualise(data_points,labels,image_name):
    """
    datapoints: np.array of shape (n,2)
    labels: np.array of shape (n,)
    """
    plt.figure(figsize=(8, 6))
    plt.scatter(data_points[:, 0], data_points[:, 1], c=labels, cmap='viridis')
    plt.colorbar()
    plt.title('Generated 2D Data from 5 Gaussian Distributions')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.savefig(image_name)
    plt.show()

def net_f1score(predictions, true_labels):
    """Calculate the multclass f1 score of the predictions.
    For this, we calculate the f1-score for each class 
    Args:
        predictions (np.array): The predicted labels.
        true_labels (np.array): The true labels.
    Returns:
        float(list): The f1 score of the predictions for each class
    """

    def precision(predictions, true_labels, label):
        """Calculate the multclass precision of the predictions.
        For this, we take the class with given label as the positive class and the rest as the negative class.
        Args:
            predictions (np.array): The predicted labels.
            true_labels (np.array): The true labels.
        Returns:
            float: The precision of the predictions.
        """
        """Start of your code."""
        """End of your code."""
        true_positives = np.sum((predictions == label) & (true_labels == label))
        false_positives = np.sum((predictions == label) & (true_labels != label))
        
        if true_positives + false_positives == 0:
            return 0.0
        else:
            return true_positives / (true_positives + false_positives)


    def recall(predictions, true_labels, label):
        """Calculate the multclass recall of the predictions.
        For this, we take the class with given label as the positive class and the rest as the negative class.
        Args:
            predictions (np.array): The predicted labels.
            true_labels (np.array): The true labels.
        Returns:
            float: The recall of the predictions.
        """
        """Start of your code."""
        """End of your code."""
        true_positives = np.sum((predictions == label) & (true_labels == label))
        false_negatives = np.sum((predictions != label) & (true_labels == label))
        
        if true_positives + false_negatives == 0:
            return 0.0
        else:
            return true_positives / (true_positives + false_negatives)
        

    def f1score(predictions, true_labels, label):
        """Calculate the f1 score using it's relation with precision and recall.
        Args:
            predictions (np.array): The predicted labels.
            true_labels (np.array): The true labels.

        Returns:
            float: The f1 score of the predictions.
        """
        precision_value = precision(predictions, true_labels, label)
        recall_value = recall(predictions, true_labels, label)
        
        if precision_value + recall_value == 0:
            return 0.0
        else:
            return (2 * precision_value * recall_value) / (precision_value + recall_value)
    

    f1s = []
    for label in np.unique(true_labels):
        f1s.append(f1score(predictions, true_labels, label))
    return f1s

def accuracy(predictions,true_labels):
    """

    You are not required to modify this part of the code.

    """
    return np.sum(predictions==true_labels)/predictions.size

if __name__ == "__main__":
    """

    You are not required to modify this part of the code.

    """

    # Load the data
    train_dataset = pd.read_csv('./data/train_dataset.csv',index_col=0).to_numpy()
    validation_dataset = pd.read_csv('./data/validation_dataset.csv',index_col=0).to_numpy()

    # Extract the data
    train_datapoints = train_dataset[:,:-1]
    train_labels = train_dataset[:, -1]
    validation_datapoints = validation_dataset[:, 0:-1]
    validation_labels = validation_dataset[:, -1]

    # Visualize the data
    # visualise(train_datapoints, train_labels, "train_data.png")

    # Train the model
    model = NaiveBayes()
    model.fit(train_datapoints, train_labels)

    # Make predictions
    train_predictions = model.predict(train_datapoints)
    validation_predictions = model.predict(validation_datapoints)

    # Calculate the accuracy
    train_accuracy = accuracy(train_predictions, train_labels)
    validation_accuracy = accuracy(validation_predictions, validation_labels)

    # Calculate the f1 score
    train_f1score = net_f1score(train_predictions, train_labels)
    validation_f1score = net_f1score(validation_predictions, validation_labels)

    # Print the results
    print('Training Accuracy: ', train_accuracy)
    print('Validation Accuracy: ', validation_accuracy)
    print('Training F1 Score: ', train_f1score)
    print('Validation F1 Score: ', validation_f1score)

    # Save the model
    save_model(model)

    # Visualize the predictions
    # visualise(validation_datapoints, validation_predictions, "validation_predictions.png")