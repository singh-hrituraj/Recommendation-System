import load_data
import Model_Trainer
import math




def check_accuracy():
    total = 0
    correct = 0
    for i in range(load_data.numstudents):
        for j in range(load_data.numcourses):
            if load_data.R[j, i] == 1:
                if math.fabs(load_data.Y[j, i] - Model_Trainer.output[j, i]) < 0.8:
                    correct = correct + 1
                total = total + 1

    print "Result is %d / %d" % (correct, total)
    accuracy = 100 * (float(correct) / total)
    print  " Accuracy : {0} '%'".format(accuracy)

    return accuracy

check_accuracy()
