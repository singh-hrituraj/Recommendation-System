import scipy.io as sio
import numpy as np


data = sio.loadmat('ex8_movies.mat') #load the movie data set

Y = np.array(data['Y'])
R = np.array(data['R'], dtype= bool)
