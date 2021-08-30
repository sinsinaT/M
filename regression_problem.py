"""univariate linear regression """

from numpy import *
from sklearn.metrics import mean_squared_error as mse
import matplotlib.pyplot as plt


def prediction_model():
    import pandas as pd
    from sklearn import linear_model
    import matplotlib.pyplot as plt

    df = pd.read_csv("ex1data1.txt", sep=",", names=["population", "profit"])

    reg = linear_model.LinearRegression()

    reg.fit(df[["population"]], df.profit)

    plt.xlabel("population")
    plt.ylabel("profit")
    plt.scatter(df["population"], df["profit"], color = 'red', marker="+")
    plt.plot(df[["population"]], reg.predict(df[["population"]]), color= "blue")
    plt.show()
    return reg.coef_, reg.intercept_


"""implementing the linear regression from scratch"""


# y = mx + b
# m is slope, b is y-intercept
def compute_error_for_line_given_points(b, m, points):
    totalError = 0
    for i in range(0, len(points)):
        x = points[i, 0]
        y = points[i, 1]
        totalError += (y - (m * x + b)) ** 2
    return totalError / float(len(points))


def step_gradient(b_current, m_current, points, learningRate):
    b_gradient = 0
    m_gradient = 0
    N = float(len(points))
    for i in range(0, len(points)):
        x = points[i, 0]
        y = points[i, 1]
        b_gradient += -(2/N) * (y - ((m_current * x) + b_current))
        m_gradient += -(2/N) * x * (y - ((m_current * x) + b_current))
    new_b = b_current - (learningRate * b_gradient)
    new_m = m_current - (learningRate * m_gradient)
    return [new_b, new_m]


def gradient_descent_runner(points, starting_b, starting_m, learning_rate, num_iterations):
    b = starting_b
    m = starting_m
    err = []
    for i in range(num_iterations):
        b, m = step_gradient(b, m, array(points), learning_rate)
        err.append(compute_error_for_line_given_points(b, m, points))
    plt.scatter(range(num_iterations), err, color='green')
    plt.show()
    return [b, m]


# def plot_regression_lines(points,m, b):
#     initial_m = m
#     initial_b = b
#
#     line1 = []
#     for i in range(len(points)):
#         line1.append(points[i,0] * initial_m + initial_b)
#
#     plt.scatter(points[:,0], points[:,1], color='red')
#     plt.plot(points[:,0], line1, color='black', label='line')
#     plt.xlabel('population')
#     plt.ylabel('profit')
#     plt.legend()
#     MSE = mse(points[:,1], line1)
#     plt.title("m value " + str(initial_m) + " with MSE " + str(MSE))


def run():
    # writing from scratch
    points = genfromtxt("ex1data1.txt", delimiter=",")
    learning_rate = 0.001
    initial_b = 0 # initial y-intercept guess
    initial_m = 0 # initial slope guess
    num_iterations = 1000
    print( "Starting gradient descent at b = {0}, m = {1}, ms error = {2}".format(initial_b, initial_m, compute_error_for_line_given_points(initial_b, initial_m, points)))
    # plot_regression_lines(points, initial_m, initial_b)
    print("Running...")
    [b, m] = gradient_descent_runner(points, initial_b, initial_m, learning_rate, num_iterations)
    print("After {0} iterations b = {1}, m = {2}, error = {3}".format(num_iterations, b, m, compute_error_for_line_given_points(b, m, points)))

    #using linear regression library
    b, m = prediction_model()
    print(m, b)

if __name__ == '__main__':
    run()