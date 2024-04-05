'''The code aims to approximate a function according to a set of data points, which is regression in this case.'''

import numpy as np
import matplotlib.pyplot as plt

# we should sample the data points from the function sinx
def sample_data(sample_num):
    x = np.linspace(0,1,sample_num) # sample 100 points from -10 to 10 randomly
    # [0.84421305, -0.28919214, -0.95003651, -0.63587365, -0.65292033, 1.00880848]
    # [-0.42198864, -0.94267747, -0.13719203, -0.33263249,  0.59505294,  0.62779226]
    print(f"the sampled x is {x}")
    y = np.sin(x*2*np.pi) # the function we want to approximate

    return x, y

def function(x, params):
    sum=0
    for i in range(len(params)):
        sum+=params[i]*(x**i)
    return sum

# we define the regression model
def regression(x, y, params_num, learning_rate, epochs, lamda):
    # although we can directly use the numpy function to solve the linear regression problem, we still implement it by ourselves
    params = np.random.normal(0,1,params_num) # initialize the weights
    #params = np.array([-0.04063483, 8.31634849, -16.38149829, -4.80867479, 4.59907882, 7.89030372, 5.11392639, 1.40958474, -0.60948956, -2.7400939, -2.44248929, -0.37625659]) # initialize the weights
    # [0.84421305, -0.28919214, -0.95003651, -0.63587365, -0.65292033, 1.00880848]
    # [-0.42198864, -0.94267747, -0.13719203, -0.33263249,  0.59505294,  0.62779226]
    # [ 0.27608881,  4.32725789, -7.4155481,  -5.27747108, -0.68728641,  3.58638638, 4.08512184,  3.3550655,   2.77465056,  0.19513605, -1.74370396, -3.4137205 ]
    # [ 0.12672685,  5.90106223, -9.95772358, -6.19495894, -0.04754629,  4.85930913, 5.31095454,  4.17424614,  3.04713103, -0.09409248, -2.5467968,  -4.65631917]
    # [-0.00284670836, 7.6464567, -14.2697698, -5.48600058, 2.74906383, 6.79602218, 5.41359434, 2.78383212, 1.07825318, -1.60153892, -2.64409838, -2.56807037]
    # [-0.04063483, 8.31634849, -16.38149829, -4.80867479, 4.59907882, 7.89030372, 5.11392639, 1.40958474, -0.60948956, -2.7400939, -2.44248929, -0.37625659]
    # [-0.06496642, 8.73185767, -17.58266374, -4.7499336, 5.85931986, 8.8951147, 5.10620246, 0.46162902, -1.94484023, -3.73623156, -2.38205461, 1.36141927]

    print(f"the initial params are {params}")

    # we use the distance loss function
    for i in range(epochs):
        
        print(f"****the {i}th round epoch****")
        # first we show the loss
        loss = 0
        for j in range(len(x)):
            loss+=(function(x[j], params)-y[j])**2/2
            # print(f"the predicted value is {function(x[j], params)} and the ground truth is {y[j]}")
        print(f"the average loss is {loss/len(x)}")
        
        # then we calculate the derivative of the loss function
        derivative = np.zeros(params_num)
        for j in range(len(x)):
            for k in range(params_num):
                derivative[k]+=(function(x[j], params)-y[j])*(x[j]**k)
        # add the regularization term   
        for j in range(params_num):
            derivative[j]+=lamda*params[j]
            
        # update the params
        for j in range(params_num):
            params[j]-=learning_rate*derivative[j]
            # print(f"the {j}th derivative is {derivative[j]}")
        print(f"the updated params are {params}")
            
    return params

def show_result(params):
    # 使用 np.poly1d 创建多项式对象
    poly_function = np.poly1d(params[::-1])

    # 定义 x 范围
    x_values = np.linspace(0, 1, 100)  # 生成 -10 到 10 之间的 100 个均匀分布的点

    # 计算对应的 y 值
    y_values = poly_function(x_values)
    y_gt = np.sin(x_values*2*np.pi)

    # 绘制多项式的图形
    plt.plot(x_values, y_gt, label='Ground Truth')
    plt.plot(x_values, y_values, label='Polynomial')
    
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Plot of Polynomial Function')
    plt.legend()
    plt.grid(True)
    plt.show()
    
def main():
    print("****start the regression****")
    sample_num = 50
    x, y = sample_data(sample_num=sample_num)
    params = regression(x, y, 16, 0.015, 10000, 0) 
    show_result(params)
                    
            
if __name__ == "__main__":
    main()