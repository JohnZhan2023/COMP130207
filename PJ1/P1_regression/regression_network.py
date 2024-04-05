'''The code aims to approximate a function according to a set of data points'''

import numpy as np
import matplotlib.pyplot as plt

# we should sample the data points from the function sinx
def sample_data(sample_num, lower_bound, upper_bound):
    x = np.linspace(lower_bound,upper_bound,sample_num)
    y = np.sin(x) # the function we want to approximate

    return x, y

def make_MLP(input, output, hidden_layer):
    W1 = np.random.randn(input, hidden_layer)
    b1 = np.random.uniform(-1, 0, (hidden_layer,))

    W2 = np.random.randn(hidden_layer, output)
    b2 = np.random.uniform(-1, 0, (output,))
    return W1, b1, W2, b2

def softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=1).reshape(-1, 1)

def sigmoid(x):
    return 1/(1+np.exp(-x))

def forward(x, W1, b1, W2, b2):
    z1 = np.dot(x, W1) + b1
    a1 = sigmoid(z1)
    z2 = np.dot(a1, W2) + b2
    return z2

def backward(x_, y_, W1, b1, W2, b2, learning_rate):
    # x y is the batch data
    dW2 = np.zeros_like(W2)
    db2 = np.zeros_like(b2)
    dW1 = np.zeros_like(W1)
    db1 = np.zeros_like(b1)
    loss = 0
    for i in range(len(x_)):
        x = x_[i].reshape(1, -1)
        y = y_[i]
        z1 = np.dot(x, W1) + b1
        a1 = sigmoid(z1)
        z2 = np.dot(a1, W2) + b2
        loss += np.sum((z2-y)**2)/2
        
        
        # calculate the derivative
        dW2 += np.dot(a1.T, (z2-y)) # (hidden_layer, output), (output, )=>(hidden_layer, output)
        db2 += ((z2-y)).flatten() # (output, )
        dW1 += np.dot(x.T, np.dot((z2-y), W2.T)*a1*(1-a1)) # (input, hidden_layer), (output, hidden_layer)=>(input, hidden_layer)
        db1 += (np.dot((z2-y), W2.T)*a1*(1-a1)).flatten() # (hidden_layer, )
    print(f"the loss is {loss/len(x_)}")
    W1 -= learning_rate*dW1
    b1 -= learning_rate*db1
    W2 -= learning_rate*dW2
    b2 -= learning_rate*db2
    
    return W1, b1, W2, b2
    
def regression(x, y,input,output, hidden_layer,batch_size ,learning_rate, epochs):
    W1, b1, W2, b2 = make_MLP(input,output , hidden_layer)
    idx = np.random.permutation(len(x))
    for i in range(epochs):
        print(f"****the {i}th round epoch****")
        for j in range(0, len(x), batch_size):
            x_ = x[idx[j:j+batch_size]]
            y_ = y[idx[j:j+batch_size]]
            W1, b1, W2, b2 = backward(x_, y_, W1, b1, W2, b2, learning_rate)
    
    
    return [W1, b1, W2, b2]
    

def show_result(params, lower_bound=0, upper_bound=1):
    x_values = np.linspace(lower_bound, upper_bound, 100) 
    y_values = np.zeros(len(x_values))
    for i in range(len(x_values)):
        x_values[i] = x_values[i].reshape(1, -1)
        y_values[i] = forward(x_values[i], params[0], params[1], params[2], params[3])
    y_gt = np.sin(x_values)


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
    sample_num = 100
    upper_bound = np.pi
    lower_bound = -np.pi
    x, y = sample_data(sample_num=sample_num, lower_bound=lower_bound, upper_bound=upper_bound)
    params = regression(x, y,1,1, 32, 5, 0.001, 1000) 
    show_result(params, lower_bound, upper_bound)
                    
            
if __name__ == "__main__":
    main()