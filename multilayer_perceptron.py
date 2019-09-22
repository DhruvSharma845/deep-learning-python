#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 22 11:45:37 2019

@author: dhruv
"""
import numpy as np

# n : number of features
# m : number of training examples
# X : (n x m) numpy array
# y : (1 x m) numpy array 
class NeuralNetTwoLP(object):
    
    def __init__(self, random_state = 1, hidden_units = 16, l2=0.0, epochs = 100, batch_size = 0, learning_rate=0.01):
        self.random = np.random.RandomState(random_state)
        self.hidden_units = hidden_units
        self.l2 = l2
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate

    def oneHot_(self, y, n_classes):
        onehot = np.zeros((n_classes, y.shape[0]))
        for idx, val in enumerate(y.astype(int)):
            onehot[val, idx] = 1.
        return onehot
    
    def sigmoid(self, Z):
        return 1. / (1. + np.exp(-np.clip(Z, -250, 250)))
    
    def forward_(self, X):
        
        # Z_h : (hidden_units x m)
        Z_h = np.dot(self.W_h, X) + self.b_h
        A_h = self.sigmoid(Z_h)
        
        Z_o = np.dot(self.W_o, A_h) + self.b_o
        A_o = self.sigmoid(Z_o)
        
        return Z_h, A_h, Z_o, A_o
    
    def compute_cost(self, y_enc, output):
        L2_term = (self.l2 * (np.sum(self.W_h ** 2.) + np.sum(self.W_o ** 2.)))
        term1 = -y_enc * (np.log(output))
        term2 = (1. - y_enc) * np.log(1. - output)
        cost = np.sum(term1 - term2) + L2_term
        return cost
        
    def predict(self, X):
        Z_h, A_h, Z_o, A_o = self.forward_(X)
        y_pred = np.argmax(A_o, axis=1)
        return y_pred
    
    def mini_batch_generator(self, X, y, batch_size):
        for index in range(0, X.shape[1], batch_size):
            yield X[:, index: index+batch_size], y[:, index: index+batch_size]
    
    def fit(self, X, y):
        
        n = X.shape[0]
        n_output = np.unique(y).shape[0]
        
        self.W_h = self.random.normal(loc=0.0, scale=0.1, size=(self.hidden_units, n))
        self.b_h = np.zeros((self.hidden_units, 1))
        
        self.W_o = self.random.normal(loc=0.0, scale=0.1, size=(n_output, self.hidden_units))
        self.b_o = np.zeros((n_output, 1))
        
        self.eval_ = {'cost': [], 'train_acc': []}
        y_enc = self.oneHot_(y, n_output)
        
        print('Shape of encoded y: ', y_enc.shape)

        
        for epoch in range(self.epochs):
            print('Iteration: ' , epoch)
            for X_batch, y_batch in self.mini_batch_generator(X, y_enc, self.batch_size):
                Z_h, A_h, Z_o, A_o = self.forward_(X_batch)
                print('Shape of Z_h: ', Z_h.shape)
                print('Shape of A_h: ', A_h.shape)
                print('Shape of Z_o: ', Z_o.shape)
                print('Shape of A_o: ', A_o.shape)
                
                # sigma_out : (n_output x m)
                sigma_out = A_o - y_batch
                dw_h = A_h * (1.0 - A_h)
                
                # element wise product
                # sigma_h : (hidden x m)
                sigma_h = (np.dot(self.W_o.T,sigma_out) * dw_h)
                
                # grad_w_h : (hidden x n)
                grad_w_h = np.dot(sigma_h, X_batch.T)
                grad_b_h = np.sum(sigma_h, axis=1).reshape((self.hidden_units, 1))
                
                # grad_w_out : (n_output, hidden)
                grad_w_out = np.dot(sigma_out, A_h.T)
                grad_b_out = np.sum(sigma_out, axis=1).reshape((n_output, 1))
                
                delta_w_h = (grad_w_h + self.l2*self.W_h)
                delta_b_h = grad_b_h # bias is not regularized

                self.W_h -= self.learning_rate * delta_w_h
                self.b_h -= self.learning_rate * delta_b_h

                delta_w_out = (grad_w_out + self.l2*self.W_o)
                delta_b_out = grad_b_out # bias is not regularized
                
                self.W_o -= self.learning_rate * delta_w_out
                self.b_o -= self.learning_rate * delta_b_out
                
            # Evaluation after each epoch during training
            z_h, a_h, z_out, a_out = self.forward_(X)
            cost = self.compute_cost(y_enc=y_enc,output=a_out)
            y_pred = self.predict(X)
            train_acc = ((np.sum(y == y_pred)).astype(np.float) / X.shape[0])
            
            self.eval_['cost'].append(cost)
            self.eval_['train_acc'].append(train_acc)
            
        return self.eval_ 
            

from tensorflow.keras.datasets import mnist
(X_train, y_train), (X_test, y_test) = mnist.load_data(path='mnist.npz')
X_train = X_train.reshape((-1,784))
X_test = X_test.reshape((-1,784))
print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)

nn_clf = NeuralNetTwoLP(hidden_units=32, batch_size=512, epochs=2)
eval = nn_clf.fit(X_train.T, y_train.T)
print('Evaluation metrics: ' , eval)
