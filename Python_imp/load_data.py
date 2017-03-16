import scipy.io as sio


data = sio.loadmat('ex8_movies.mat') #load the movie data set

Y = data['Y']
R = data['R']


Parameters = sio.loadmat('ex8_movieParams.mat')

X = Parameters['X']

Theta = Parameters['Theta']

numstudents = Parameters['num_users']
numfeatures = Parameters['num_features']
numcourses = Parameters['num_movies']



