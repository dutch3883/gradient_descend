import matplotlib.pyplot as plt
import matplotlib.animation as animation
import time
import math

def get_data():
    points =[]
    with open('data.csv','r') as file:
        for i in file.readlines():
            points.append({'x':float(i.split(',')[0]),'y': float(i.split(',')[1])})
            points = sorted(points,key=lambda a:a['x'])
    ret_dat = []
    for i in points:
        ret_dat.append([i['x'],i['y']])
    print(ret_dat)
    return ret_dat



ax1=None
fig=None

def ploter(f,data):
    time.sleep(0.001)
    xs = [i[0] for i in data]
    ys = [i[1] for i in data]

    global  ax1
    global fig
    forecast_line_y = []
    first_time = False
    for x in xs:
        forecast_line_y.append(f(x))
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

def cal_a_point_error(point,f):
    x = point[0]
    y = point[1]
    error = (y - f(x))**2
    return error

def cal_avg_error_for_points(points,f):
    total_error =0
    for i in points:
        total_error+=cal_a_point_error(i,f)
    return total_error/len(points)

def partial_diff_quadratic(points,theta,f,learning_rate):
    N = len(points)
    diff_theta = [0]*len(theta)
    for i in points:
        x = i[0]
        y = i[1]
        for i in range(0,len(diff_theta)):
            diff_theta[i] += -2 / N * (y - (f(x))) * math.pow(x,i)

    for i in range(0,len(theta)):
        theta[i] = theta[i] - diff_theta[i] * learning_rate
    return theta

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

def descent_gradient(data,dim,iteration):
    learning_rate = 0.0001
    theta = [0.030269517287775907, 1.478895662279729]
    error =0

    def fx(x):
        accum = 0
        for i in range(0, len(theta)):
            accum += theta[i] * math.pow(x, i)
        return accum

    for i in range(iteration):
        theta = partial_diff_quadratic(data,theta,fx,learning_rate=learning_rate)

        def fx(x):
            accum =0
            for i in range(0,len(theta)):
                accum += theta[i]*math.pow(x,i)
            return accum

        new_error = cal_avg_error_for_points(data,fx)
        if error != 0 and new_error -error > 0:
            learning_rate/=10
        print('iteration {} : error = {} change = {} learning rate = {}'.format(i+1,new_error,new_error-error,learning_rate))

        ploter(fx,data)
        error = new_error
    return theta

if __name__  == "__main__":
    g = get_data()
    print(descent_gradient(g,2,20))