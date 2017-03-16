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


def loadcoursesList():

    with open("course_ids.txt") as course_file:
        course_list = []

        for i,course in enumerate(course_file.readlines()):
            course_name  = course.split()[1:]
            course_list.append(" ".join(course_name))

    return course_list
  
loadcoursesList()
