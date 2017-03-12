import load_data
import numpy as np

X = load_data.X
Y = load_data.Y
R = load_data.R
Theta = load_data.Theta
numusers = load_data.numusers
nummovies = load_data.nummovies
numfeatures = load_data.numfeatures


def costfunction(X, Theta, Y, R, numusers, nummovies,numfeatures, reg = 0):

    X = X[0:nummovies,0:numfeatures]
    print X
    Y = Y[0:nummovies,0:numusers]
    Theta = Theta[0:numusers, 0:numfeatures]
    R = R[0:nummovies, 0:numusers]
    print R

    Output = np.dot(X, np.transpose(Theta))

    J = 0.5 * np.sum((Output[R]- Y[R])*(Output[R] - Y[R])) + reg * np.sum(Theta *Theta) + reg * np.sum(X * X)
    return J

costfunction(X,Theta, Y, R, numusers, nummovies, numfeatures)


