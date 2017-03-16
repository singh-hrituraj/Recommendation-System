import load_data
import Costfunction as ccf
from scipy.optimize import minimize
import numpy as np

Y = load_data.Y
R = load_data.R

numstudents = load_data.numstudents
numcourses = load_data.numcourses
numfeatures = load_data.numfeatures
reg = 0

Y = Y[:numcourses,:numstudents]
R = R[:numcourses,:numstudents]



X = np.random.randn(numcourses, numfeatures)
Theta = np.random.randn(numstudents, numfeatures)
initial_parameters = np.concatenate((X.reshape(X.size, order='F'), Theta.reshape(Theta.size, order='F')))

def J(initial_parameters):
    return ccf.costfunction(initial_parameters, Y, R, numstudents, numcourses, numfeatures, reg)

maxiter = 100
options = {'disp': True, 'maxiter':maxiter}

result = minimize(J, x0= initial_parameters, options=options, method="L-BFGS-B", jac=True )

params = result['x']
X = np.reshape(params[:numcourses * numfeatures], (numcourses, numfeatures), order='F')
Theta = np.reshape(params[numcourses * numfeatures:], (numstudents, numfeatures), order='F')
output = np.dot(X, Theta.T)



