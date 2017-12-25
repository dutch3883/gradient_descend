import matplotlib.pyplot as plt
import matplotlib.animation as animation
import time


def get_data():
    points =[]
    with open('data.csv','r') as file:
        for i in file.readlines():
            points.append([float(x) for x in i.split(',')])
    return points
ax1=None
fig=None
def ploter(m,b,data):
    time.sleep(0.001)
    xs = [i[0] for i in data]
    ys = [i[1] for i in data]
    global  ax1
    global fig
    forecast_line_y = []
    first_time = False
    for x in xs:
        forecast_line_y.append(m*x+b)
    if ax1 is None:

        first_time =True
        fig = plt.figure()
        ax1 = fig.gca()
    else:
        ax1.clear()
    ax1.plot(xs, forecast_line_y, label='predict')
    ax1.plot(xs, ys, marker='o', linestyle=' ', color='r', label='data')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('regression')
    if first_time:
        plt.show(block=False)
    else:
        fig.canvas.draw()

def cal_a_point_error(point,m,b):
    x = point[0]
    y = point[1]
    error = (y - (m*x+b))**2
    return error

def cal_avg_error_for_points(points,m,b):
    total_error =0
    for i in points:
        total_error+=cal_a_point_error(i,m,b)
    return total_error/len(points)

def partial_diff_m(points,m,b,learning_rate):
    N =  len(points)
    diff_m = 0
    diff_b = 0
    for i in points:
        x = i[0]
        y = i[1]
        diff_b += -2 / N * (y - (m * x + b))
        diff_m += -2 * x / N * (y - (m * x + b))
    new_m = m - diff_m*learning_rate
    new_b  = b - diff_b*learning_rate
    return new_m, new_b

def descent_gradient(data,init_m,init_b,iteration):
    b= init_b
    m = init_m
    learning_rate = 1
    error = 0
    for i in range(iteration):
        m,b = partial_diff_m(data,m=m,b=b,learning_rate=learning_rate)
        new_error = cal_avg_error_for_points(data,m=m,b=b)
        if error != 0 and new_error -error > 0:
            learning_rate/=5
        print('iteration {} : error = {} change = {} learning rate = {}'.format(i+1,new_error,new_error-error,learning_rate))
        ploter(m,b,data)
        error = new_error


if __name__  == "__main__":
    g = get_data()
    descent_gradient(g,1,0,10000)