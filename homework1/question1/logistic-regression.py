#!/usr/bin/env python

# Deep Learning Homework 1

import argparse
import time
import pickle
import json

import numpy as np

import utils

class PCA:
    def __init__(self, n_components=100):
        self.n_components = n_components
        self.components = None
        self.mean = None
        self.explained_variance = None
    
    def fit(self, X):
        """
        Fit PCA on training data
        X: (n_samples, n_features)
        """
        # Center the data
        self.mean = np.mean(X, axis=0)
        X_centered = X - self.mean
        
        # Compute covariance matrix
        cov = np.cov(X_centered.T)
        
        # Eigendecomposition
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        
        # Sort by eigenvalues (descending order)
        idx = eigenvalues.argsort()[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        
        # Select top n_components
        self.components = eigenvectors[:, :self.n_components]
        self.explained_variance = eigenvalues[:self.n_components]
        
        return self
    
    def transform(self, X):
        """
        Transform data using fitted PCA
        X: (n_samples, n_features)
        returns: (n_samples, n_components)
        """
        X_centered = X - self.mean
        return X_centered @ self.components
    
    def fit_transform(self, X):
        """
        Fit and transform in one step
        """
        self.fit(X)
        return self.transform(X)


class LogisticRegression:
    def __init__(self, n_classes, n_features, eta, l2_penalty):
        self.W = np.zeros((n_classes, n_features))
        self.eta = eta
        self.l2penalty = l2_penalty
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
        return exp_z / np.sum(exp_z, axis=0, keepdims=True)
    
    def update_weight(self, x_i, y_i):
        """
        x_i (n_features,): a single training example
        y_i (scalar): the gold label for that example
        """

        x_i = np.expand_dims(x_i, axis=1)  # (p+1, 1)
        z = self.W.dot(x_i)  # (n_classes, 1)
        
        # Softmax probabilities
        p = self.softmax(z = z)  # (n_classes, 1)
        
        # One-hot encode true label
        y_one_hot = np.zeros((self.n_classes, 1))
        y_one_hot[y_i] = 1
        
        # SGD update
        gradient = (p - y_one_hot).dot(x_i.T)  # (n_classes, p+1)

        #update the weights with weight decay

        self.W = (1 - self.eta * self.l2penalty) * self.W - self.eta * gradient

    def train_epoch(self, X, y):
        """
        X (n_examples, n_features): features for the whole dataset
        y (n_examples,): labels for the whole dataset
        One epoch of SGD for multi-class logistic regression
        """
        
        for x_i, y_i in zip(X, y):
                self.update_weight(x_i=x_i, y_i=y_i)
               
            

    def predict(self, X):
        """
        X (n_examples, n_features)
        returns predicted labels y_hat, whose shape is (n_examples,)
        """
    
        # Compute probabilities for all samples
        z = X.dot(self.W.T)  # (n, n_classes)
        
        # Get predicted class (argmax)
        y_hat = np.argmax(z, axis=1)
        
        return y_hat

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


def perform_grid_search(args, data):
    print("Starting grid search...")
    #original data
    X_train_original, y_train = data["train"]
    X_valid_original, y_valid = data["dev"]
    X_test_original, y_test = data["test"]


    #different feature representation, in this case PCA
    print(f'Applying PCA with {args.n_components} components...')
    print(f'Original feature dimension: {X_train_original.shape[1]}')
    
    # Remove bias column before PCA, add it back after
    # Assuming last column is bias (all ones)
    X_train_no_bias = X_train_original[:, :-1]
    X_valid_no_bias = X_valid_original[:, :-1]
    X_test_no_bias = X_test_original[:, :-1]
    
    # Fit PCA on training data
    pca = PCA(n_components=args.n_components)
    X_train_pca = pca.fit_transform(X_train_no_bias)
    X_valid_pca = pca.transform(X_valid_no_bias)
    X_test_pca = pca.transform(X_test_no_bias)
    
    # Add bias column back
    X_train_pca = np.column_stack([X_train_pca, np.ones(X_train_pca.shape[0])])
    X_valid_pca = np.column_stack([X_valid_pca, np.ones(X_valid_pca.shape[0])])
    X_test_pca = np.column_stack([X_test_pca, np.ones(X_test_pca.shape[0])])
    
    print(f'PCA feature dimension: {X_train_pca.shape[1]}')
    
    # Print explained variance
    explained_var_ratio = pca.explained_variance / np.sum(pca.explained_variance)
    print(f'Explained variance ratio: {np.sum(explained_var_ratio):.4f}')

    etas_grid = [0.0001, 0.001, 0.01]
    l2_penalty_grid = [0.00001, 0.00005, 0.0001]  
    representation_grid = ['original', 'alternative']

    results = []

    for representation in representation_grid:
        for eta in etas_grid:
            for l2_penalty in l2_penalty_grid:
                print(f"Starting Experiment- feature representation: {representation} - eta: {str(eta)} - l2_penalty: {str(l2_penalty)}")

                if representation == 'original':
                    X_train = X_train_original
                    X_valid = X_valid_original
                    X_test = X_test_original
                elif representation == 'alternative': 
                    X_train = X_train_pca
                    X_valid = X_valid_pca
                    X_test = X_test_pca

                n_classes = np.unique(y_train).size
                n_feats = X_train.shape[1]

                # initialize the model
                model = LogisticRegression(n_classes, n_feats, eta=eta, l2_penalty=l2_penalty)

                epochs = np.arange(1, args.epochs + 1)

                valid_accs = []
                train_accs = []

                start = time.time()

                best_valid = 0.0
                best_epoch = -1
                for i in epochs:
                    print('Training epoch {}'.format(i))
                    train_order = np.random.permutation(X_train.shape[0])

                    model.train_epoch(X_train[train_order], y_train[train_order])

                    train_acc = model.evaluate(X_train[train_order], y_train[train_order])
                    valid_acc = model.evaluate(X_valid, y_valid)

                    train_accs.append(train_acc)
                    valid_accs.append(valid_acc)

                    print('train acc: {:.4f} | val acc: {:.4f}'.format(train_acc, valid_acc))
                    

                    if(valid_acc > best_valid):
                        # print("new checkpoint")
                        best_valid = valid_acc
                        best_epoch = i
                        model.save(args.save_path)

                elapsed_time = time.time() - start
                minutes = int(elapsed_time // 60)
                seconds = int(elapsed_time % 60)
                print('Training took {} minutes and {} seconds'.format(minutes, seconds))

                print("Reloading best checkpoint")
                best_model = LogisticRegression.load(args.save_path)
                test_acc = best_model.evaluate(X_test, y_test)

                print('Best model test acc: {:.4f}'.format(test_acc))

                results_obj = {
                    "representation": representation,
                    "eta": eta,
                    "l2_penalty": l2_penalty,
                    "val_acc": best_valid,
                    "test_acc": test_acc
                }
                results.append(results_obj)

                del model
            
    print("Grid search complete - Results:")

    #find the best config
    results.sort(key=lambda x: x["test_acc"], reverse=True)

    print(f"BEST CONFIG: representation: {results[0]['representation']} - eta: {str(results[0]['eta'])} - l2_penalty: {str(results[0]['l2_penalty'])} - val_acc: {str(results[0]['val_acc'])} - test_acc: {str(results[0]['test_acc'])}")
                
            




def main(args):
    utils.configure_seed(seed=args.seed)

    data = utils.load_dataset(data_path=args.data_path, bias=True)
    X_train, y_train = data["train"]
    X_valid, y_valid = data["dev"]
    X_test, y_test = data["test"]

    
    if args.grid_search:
        perform_grid_search(args, data)
        return


    # Apply PCA if requested
    if args.use_pca:
        print(f'Applying PCA with {args.n_components} components...')
        print(f'Original feature dimension: {X_train.shape[1]}')
        
        # Remove bias column before PCA, add it back after
        # Assuming last column is bias (all ones)
        X_train_no_bias = X_train[:, :-1]
        X_valid_no_bias = X_valid[:, :-1]
        X_test_no_bias = X_test[:, :-1]
        
        # Fit PCA on training data
        pca = PCA(n_components=args.n_components)
        X_train_pca = pca.fit_transform(X_train_no_bias)
        X_valid_pca = pca.transform(X_valid_no_bias)
        X_test_pca = pca.transform(X_test_no_bias)
        
        # Add bias column back
        X_train = np.column_stack([X_train_pca, np.ones(X_train_pca.shape[0])])
        X_valid = np.column_stack([X_valid_pca, np.ones(X_valid_pca.shape[0])])
        X_test = np.column_stack([X_test_pca, np.ones(X_test_pca.shape[0])])
        
        print(f'New feature dimension: {X_train.shape[1]}')
        
        # Print explained variance
        explained_var_ratio = pca.explained_variance / np.sum(pca.explained_variance)
        print(f'Explained variance ratio: {np.sum(explained_var_ratio):.4f}')

    n_classes = np.unique(y_train).size
    n_feats = X_train.shape[1]

    # initialize the model
    model = LogisticRegression(n_classes, n_feats, eta=0.0001, l2_penalty=0.00001)

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
    best_model = LogisticRegression.load(args.save_path)
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
    parser.add_argument("--use-pca", action="store_true", 
                        help="Apply PCA feature transformation")
    parser.add_argument("--n-components", type=int, default=100,
                        help="Number of PCA components to keep")
    parser.add_argument("--grid-search", action="store_true")

    args = parser.parse_args()
    main(args)