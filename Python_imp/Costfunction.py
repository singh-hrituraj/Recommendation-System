import numpy as np


def costfunction(params,Y, R, num_students, num_courses, num_features, lambda_var):


    X = np.reshape(params[:num_courses * num_features], (num_courses, num_features), order='F')
    Theta = np.reshape(params[num_courses * num_features:], (num_students, num_features), order='F')

    squared_error = np.power(np.dot(X, Theta.T) - Y, 2)
    J = (1 / 2.) * np.sum(squared_error * R)
    J = J + (lambda_var / 2.) * (np.sum(np.power(Theta, 2)) + np.sum(np.power(X, 2)))

    X_grad = np.dot((np.dot(X, Theta.T) - Y) * R, Theta)
    Theta_grad = np.dot(((np.dot(X, Theta.T) - Y) * R).T, X)

    X_grad = X_grad + lambda_var * X
    Theta_grad = Theta_grad + lambda_var * Theta

    grad = np.concatenate((X_grad.reshape(X_grad.size, order='F'), Theta_grad.reshape(Theta_grad.size, order='F')))

    return J, grad
