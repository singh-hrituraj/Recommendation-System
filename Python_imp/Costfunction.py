import load_data
import numpy as np
X = load_data.X
Y = load_data.Y
R = load_data.R
Theta = load_data.Theta
numusers = load_data.numusers
nummovies = load_data.nummovies
numfeatures = load_data.numfeatures

X = X[0:nummovies, 0:numfeatures]
Y = Y[0:nummovies,0:numusers]
R = R[0:nummovies,0:numusers]
Theta = Theta[0:numusers,0:numfeatures]


params = np.concatenate((Theta.reshape(np.prod(Theta.shape), 1), X.reshape(np.prod(X.shape), 1)))


def costfunction(Params, *args):
    Y, R, numusers, nummovies,numfeatures, reg = args

    Theta = np.reshape(params[0:numusers * numfeatures, :], (numusers, numfeatures))
    X = np.reshape(params[numusers*numfeatures: ,:], (nummovies, numfeatures))
    

    Output = np.dot(X, np.transpose(Theta))

    J = 0.5 * np.sum((Output[R]- Y[R])*(Output[R] - Y[R])) + reg * np.sum(Theta *Theta) + reg * np.sum(X * X)
    return J

def gradientfunc(params, *args):
    Y, R, numusers, nummovies,numfeatures, reg = args
    Theta = np.reshape(params[0:numusers * numfeatures, :], (numusers, numfeatures))
    X = np.reshape(params[numusers * numfeatures:, :], (nummovies, numfeatures))

    Output = np.dot(X, np.transpose(Theta))

    Grad_X = np.dot(R * Output - R * Y, Theta) + reg * X
    Grad_Theta = np.dot(np.transpose(R * Output - R * Y), X) + reg * Theta

    Gradient = np.concatenate((Grad_Theta.reshape(np.prod(Grad_Theta.shape), 1), Grad_X.reshape(np.prod(Grad_X.shape), 1)))

    return Gradient


