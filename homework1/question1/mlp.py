#!/usr/bin/env python

# Deep Learning Homework 1

import argparse
import time
import pickle
import json

import numpy as np

import utils




class MLP:
    def __init__(self, n_classes, n_features, hidden_layer_size, eta):
        mu = 0.1
        sigma = 0.1  # Note: σ² = 0.1² = 0.01, so σ = 0.1
        
        self.W1 = np.random.normal(mu, sigma, (hidden_layer_size, n_features))
        self.b1 = np.zeros((hidden_layer_size, 1))

        self.W2 = np.random.normal(mu, sigma, (n_classes, hidden_layer_size))
        self.b2 = np.zeros((n_classes, 1))


        self.eta = eta
        self.n_classes = n_classes

    def save(self, path):
        """
        Save perceptron to the provided path
        """
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path):
        """
        Load perceptron from the provided path
        """
        with open(path, "rb") as f:
            return pickle.load(f)
        
    # Softmax function
    def softmax(self, z):
        exp_z = np.exp(z)
        return exp_z / np.sum(exp_z)
    
    def update_weight(self, grad_w2, grad_b2, grad_w1, grad_b1):
      
        self.W2 = self.W2 - self.eta * grad_w2

        self.b2 = self.b2 - self.eta * grad_b2

        self.W1 = self.W1 - self.eta * grad_w1

        self.b1 = self.b1 - self.eta * grad_b1 


          
        return
       

    def train_epoch(self, X, y):
        """
        X (n_examples, n_features): features for the whole dataset
        y (n_examples,): labels for the whole dataset
        One epoch of SGD for multi-class logistic regression
        """
        
        for x_i, y_i in zip(X, y):
                
                out, h1, z1 = self.foward_pass(X=x_i)

               
                grad_w2, grad_b2, grad_w1, grad_b1= self.backpropagation(out=out, y_true=y_i, h1=h1, z1=z1, x = x_i)

                self.update_weight(grad_w2, grad_b2, grad_w1, grad_b1)


              
        
               

    def ReLU(self, x):
        return np.maximum(0.0, x)
    
    def grad_ReLU(self, x):
        return (x > 0).astype(float)

    def foward_pass(self,X):

        # Reshape to column vector
        x = X.reshape(-1, 1)  # (n_features, 1)

        #hidden layer
        z1 = self.W1.dot(x) + self.b1       # (hidden_size, 1)
        
        h1 = self.ReLU(z1)                  # (hidden_size, 1)

        #second layer
        z2 = self.W2.dot(h1) + self.b2

        out = self.softmax(z2)

        return out, h1, z1

    def backpropagation(self, out, y_true, h1, z1, x):

        # One-hot encode true label
        y_one_hot = np.zeros((self.n_classes, 1))
        y_one_hot[y_true] = 1

        grad_z2 = out - y_one_hot

        grad_w2 = grad_z2.dot(h1.T)

        grad_b2 = grad_z2

        grad_h1 = self.W2.T.dot(grad_z2)

        grad_z1 = np.multiply(grad_h1, self.grad_ReLU(z1))

        x = x.reshape(-1, 1)
        grad_w1 = grad_z1.dot(x.T)

        grad_b1 = grad_z1

        return grad_w2, grad_b2, grad_w1, grad_b1

        

    def predict(self, X):
        """
        X (n_examples, n_features)
        returns predicted labels y_hat, whose shape is (n_examples,)
        """
    
        # Compute probabilities for all samples
        predictions = []
        for x_i in X:
            out, _, _ = self.foward_pass(X=x_i)
            predictions.append(np.argmax(out))
        
        return np.array(predictions)

    def evaluate(self, X, y):
        """
        X (n_examples x n_features)
        y (n_examples): gold labels

        returns classifier accuracy
        """
        total = len(y)
        mistake = 0

        predicted_labels = self.predict(X=X)
        for i in range(total):
            if predicted_labels[i] != y[i]:
                mistake += 1
        correct = total - mistake
        return correct / total




def main(args):
    utils.configure_seed(seed=args.seed)

    data = utils.load_dataset(data_path=args.data_path, bias=True)
    X_train, y_train = data["train"]
    X_valid, y_valid = data["dev"]
    X_test, y_test = data["test"]


    n_classes = np.unique(y_train).size
    n_feats = X_train.shape[1]

    # initialize the model
    model = MLP(n_classes, n_feats, hidden_layer_size=100,eta=0.001)

    epochs = np.arange(1, args.epochs + 1)

    valid_accs = []
    train_accs = []

    start = time.time()

    best_valid = 0.0
    best_epoch = -1
    for i in epochs:
        print('Training epoch {}'.format(i))
        train_order = np.random.permutation(X_train.shape[0])
        X_train = X_train[train_order]
        y_train = y_train[train_order]

        model.train_epoch(X_train, y_train)

        train_acc = model.evaluate(X_train, y_train)
        valid_acc = model.evaluate(X_valid, y_valid)

        train_accs.append(train_acc)
        valid_accs.append(valid_acc)

        print('train acc: {:.4f} | val acc: {:.4f}'.format(train_acc, valid_acc))
        

        if(valid_acc > best_valid):
            print("new checkpoint")
            best_valid = valid_acc
            best_epoch = i
            model.save(args.save_path)

    elapsed_time = time.time() - start
    minutes = int(elapsed_time // 60)
    seconds = int(elapsed_time % 60)
    print('Training took {} minutes and {} seconds'.format(minutes, seconds))

    print("Reloading best checkpoint")
    best_model = MLP.load(args.save_path)
    test_acc = best_model.evaluate(X_test, y_test)

    print('Best model test acc: {:.4f}'.format(test_acc))

    utils.plot(
        "Epoch", "Accuracy",
        {"train": (epochs, train_accs), "valid": (epochs, valid_accs)},
        filename=args.accuracy_plot
    )

    with open(args.scores, "w") as f:
        json.dump(
            {"best_valid": float(best_valid),
             "selected_epoch": int(best_epoch),
             "test": float(test_acc),
             "time": elapsed_time},
            f,
            indent=4
        )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', default=20, type=int,
                        help="""Number of epochs to train for.""")
    parser.add_argument('--data-path', type=str, default="emnist-letters.npz")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-path", required=True)
    parser.add_argument("--accuracy-plot", default="Q2-logreg-accs2.pdf")
    parser.add_argument("--scores", default="Q2-logreg-scores2.json")

    args = parser.parse_args()
    main(args)