import Costfunction


params = Costfunction.params

Y = Costfunction.Y
R = Costfunction.R
numusers = Costfunction.numusers
nummovies = Costfunction.nummovies
numfeatures = Costfunction.numfeatures
reg = 0

learning_rate = 0.003

iter = 10000
for i in range(iter):
    print " iter %d \n " %i

    params = params - learning_rate * Costfunction.gradientfunc(params, Y, R, numusers, nummovies, numfeatures, reg)

    print Costfunction.costfunction(params, Y, R, numusers, nummovies, numfeatures, reg)




