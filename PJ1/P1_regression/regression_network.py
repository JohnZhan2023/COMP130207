'''The code aims to approximate a function according to a set of data points'''

import numpy as np
import matplotlib.pyplot as plt

# we should sample the data points from the function sinx
def sample_data(sample_num, lower_bound, upper_bound):
    x = np.linspace(lower_bound,upper_bound,sample_num)
    y = np.sin(x) # the function we want to approximate
    x=x.reshape(-1,1)
    y=y.reshape(-1,1)

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
    print(f"the batch size is {len(x_)}")
    dW2 = np.zeros_like(W2)
    db2 = np.zeros_like(b2)
    dW1 = np.zeros_like(W1)
    db1 = np.zeros_like(b1)
    loss = 0

    x = x_
    y = y_
    z1 = np.dot(x, W1) + b1
    a1 = sigmoid(z1)
    z2 = np.dot(a1, W2) + b2
    loss += np.sum((z2-y)**2)/2
    
    
    # calculate the derivative
    dW2 += np.dot(a1.T, (z2-y)) # (hidden_layer, output), (output, )=>(hidden_layer, output)
    db2 += np.sum(((z2-y)),axis=0)# (output, )
    dW1 += np.dot(x.T, np.dot((z2-y), W2.T)*a1*(1-a1)) # (input, hidden_layer), (output, hidden_layer)=>(input, hidden_layer)
    db1 += np.sum((np.dot((z2-y), W2.T)*a1*(1-a1)), axis=0) # (hidden_layer, )
    print(f"the loss is {loss/len(x_)}")
    W1 -= learning_rate*dW1
    b1 -= learning_rate*db1
    W2 -= learning_rate*dW2
    b2 -= learning_rate*db2
    
    return W1, b1, W2, b2, loss/len(x_)
    
def regression(x, y,input,output, hidden_layer,batch_size ,learning_rate, epochs):
    W1, b1, W2, b2 = make_MLP(input,output , hidden_layer)
    idx = np.random.permutation(len(x))
    loss=[]
    
    for i in range(epochs):
        batch_loss=[]
        print(f"****the {i}th round epoch****")
        for j in range(0, len(x), batch_size):
            x_ = x[idx[j:j+batch_size]]
            y_ = y[idx[j:j+batch_size]]
            W1, b1, W2, b2, loss_ = backward(x_, y_, W1, b1, W2, b2, learning_rate)
            batch_loss.append(loss_)
        loss.append(np.mean(batch_loss))
    
    return [W1, b1, W2, b2], loss
    

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
  
def plot_losses(loss_lists, labels, params_str):
    for loss_list, label in zip(loss_lists, labels):
        plt.plot(loss_list, marker='o', linestyle='-', label=label)
    plt.xlabel('Batch')
    plt.ylabel('Loss')
    plt.title('Loss Variation')
    plt.legend()
    plt.grid(True)
    plt.text(0.5, 0.5, params_str, fontsize=10, transform=plt.gca().transAxes, bbox=dict(facecolor='white', alpha=0.5))
    plt.show()

def main():
    print("****start the regression****")
    sample_num = 100
    upper_bound = np.pi
    lower_bound = -np.pi
    x, y = sample_data(sample_num=sample_num, lower_bound=lower_bound, upper_bound=upper_bound)
    params1, loss1 = regression(x, y,1,1, 32, 5, 0.001, 200) 
    show_result(params1, lower_bound, upper_bound)
    params2, loss2 = regression(x, y,1,1, 32, 5, 0.001, 1000) 
    show_result(params2, lower_bound, upper_bound)
    params3, loss3 = regression(x, y,1,1, 32, 5, 0.001, 4000) 
    show_result(params3, lower_bound, upper_bound)
    labels = ['epochs=200', 'epochs=1000', 'epochs=4000']
    params_str = f"batch_size=5, hidden_layer=32, lr=0.001"
    plot_losses([loss1[:100], loss2[:100], loss3[:100]], labels, params_str)
    
                    
            
if __name__ == "__main__":
    main()