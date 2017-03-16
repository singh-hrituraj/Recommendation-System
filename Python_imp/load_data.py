import scipy.io as sio


data = sio.loadmat('courses_data.mat') #load the courses data set

Y = data['Y']
R = data['R']


Parameters = sio.loadmat('course_parameters.mat')

X = Parameters['X']

Theta = Parameters['Theta']

numstudents = Parameters['num_students']
numfeatures = Parameters['num_features']
numcourses = Parameters['num_courses']



