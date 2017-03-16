import load_data
from scipy.optimize import minimize
import math

import numpy as np


Y = load_data.Y
R = load_data.R

numusers = load_data.numusers
nummovies = load_data.nummovies
numfeatures = load_data.numfeatures
reg = 0

Y = Y[:nummovies,:numusers]
R = R[:nummovies,:numusers]



X = np.random.randn(nummovies, numfeatures)
Theta = np.random.randn(numusers, numfeatures)
initial_parameters = np.concatenate((X.reshape(X.size, order='F'), Theta.reshape(Theta.size, order='F')))

def costfunction(params,Y, R, num_users, num_movies, num_features, lambda_var):


    X = np.reshape(params[:num_movies * num_features], (num_movies, num_features), order='F')
    Theta = np.reshape(params[num_movies * num_features:], (num_users, num_features), order='F')






    squared_error = np.power(np.dot(X, Theta.T) - Y, 2)
    J = (1 / 2.) * np.sum(squared_error * R)
    J = J + (lambda_var / 2.) * (np.sum(np.power(Theta, 2)) + np.sum(np.power(X, 2)))
    X_grad = np.dot((np.dot(X, Theta.T) - Y) * R, Theta)
    Theta_grad = np.dot(((np.dot(X, Theta.T) - Y) * R).T, X)

    X_grad = X_grad + lambda_var * X
    Theta_grad = Theta_grad + lambda_var * Theta

    grad = np.concatenate((X_grad.reshape(X_grad.size, order='F'), Theta_grad.reshape(Theta_grad.size, order='F')))

    return J, grad


def Junction(initial_parameters):

    return costfunction(initial_parameters, Y, R, numusers, nummovies, numfeatures, reg)


maxiter = 1000
options = {'disp': True, 'maxiter':maxiter}

result = minimize(Junction, x0= initial_parameters, options=options, method="L-BFGS-B", jac=True )

params = result['x']
X = np.reshape(params[:nummovies * numfeatures], (nummovies, numfeatures), order='F')
Theta = np.reshape(params[nummovies * numfeatures:], (numusers, numfeatures), order='F')
output = np.dot(X, Theta.T)

total = 0
correct = 0
for i in range(numusers):
    for j in range(nummovies):
        if R[j,i] ==1:
            if math.fabs(Y[j,i] - output[j,i]) < 0.8:
                correct = correct + 1
            total = total + 1


print "Result is %d / %d"%(correct,total)
accuracy = 100* (float(correct) / total)
print  " Accuracy : {0} '%'".format(accuracy)



