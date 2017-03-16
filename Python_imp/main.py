import load_data
import numpy as np
import normalizeRatings
import Model_Trainer
numcourses = load_data.numcourses
courseslist = load_data.loadcoursesList()


new_user = np.zeros((load_data.Y.shape[0],1))

new_user[0] = 5
new_user[11] = 5
new_user[63] = 4
new_user[69] = 3

for i, rating in enumerate(new_user):
    if rating > 0:
        print('Rated {:.0f} for {:s}\n'.format(rating[0], courseslist[i]))

print new_user.shape
print load_data.Y.shape

load_data.Y = np.column_stack((new_user, load_data.Y))
load_data.R = np.column_stack(((new_user != 0).astype(int), load_data.R))

[load_data.Y, Y_Mean] = normalizeRatings.normalizeRatings(load_data.Y,load_data.R)


load_data.numstudents = load_data.Y.shape[1]


print "Training our model: \n Please keep patience \n"

output = Model_Trainer.output[:,0] + Y_Mean.flatten()

outputarg = output.argsort()[::-1]

for i in xrange(10):
    j = outputarg[i]
    print('Predicting rating {:.5f} for course {:s}'.format(output[j], courseslist[j]))





