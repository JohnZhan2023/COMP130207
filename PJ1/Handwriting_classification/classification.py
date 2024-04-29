'''The code aims to classify the handwritten words, which is classification in this case.'''
import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image

category=12
import numpy as np

def load_data():
    x=[]
    y=[]
    x_test=[]
    y_test=[]
    
    for i in range(category):
        root = f'../train/{i+1}'
        filenames = os.listdir(root)
        np.random.shuffle(filenames)  
        test_num = 0
        for filename in filenames:
            img_root = os.path.join(root, filename)
            image = Image.open(img_root)
            if test_num < (int)(0.2*len(filenames)):
                x_test.append(np.array(image).flatten())
                y_test.append([1 if j == i else 0 for j in range(12)])
                test_num += 1
            else:
                x.append(np.array(image).flatten())
                y.append([1 if j == i else 0 for j in range(12)])
    
    x = np.array(x)
    y = np.array(y)
    x_test = np.array(x_test)
    y_test = np.array(y_test)
    
    return x, y, x_test, y_test

    
def softmax(x):

    return np.exp(x) / np.sum(np.exp(x), axis=1).reshape(-1, 1)
def sigmoid(x):
    return 1/(1+np.exp(-x))

def make_MLP(input, output, hidden_layer):
    W1 = np.random.randn(input, hidden_layer)/np.sqrt(input*hidden_layer)
    b1 = np.random.randn(hidden_layer)/np.sqrt(hidden_layer)
    W2 = np.random.randn(hidden_layer, output)/np.sqrt(hidden_layer*output)
    b2 = np.random.randn(output)/np.sqrt(output)
    return W1, b1, W2, b2

def forward(x, W1, b1, W2, b2):
    z1 = np.dot(x, W1) + b1
    a1 = sigmoid(z1)
    z2 = np.dot(a1, W2) + b2
    a2 = softmax(z2)
    return a2



def backward(x_, y_, W1, b1, W2, b2, learning_rate):
    # x y is the batch data
    dW2 = np.zeros_like(W2)  # (hidden_layer, output)
    db2 = np.zeros_like(b2)  # (output)
    dW1 = np.zeros_like(W1)  # (input, hidden_layer)
    db1 = np.zeros_like(b1)  # (hidden_layer)

    loss = 0
    x = x_
    y = y_
    z1 = np.dot(x, W1) + b1
    a1 = sigmoid(z1)
    z2 = np.dot(a1, W2) + b2
    a2 = softmax(z2)
    loss += np.sum(-y*np.log(a2))
    # calculate the derivative
    # print(f"the shape of a1 is {a1.shape}")
    # print(f"the shape of W2 is {W2.shape}")
    # print(f"the shape of a2 is {a2.shape}")
    # print(f"the shape of y is {y.shape}")
    dW2 += np.dot(a1.T, a2-y) # (batch_size,hidden_layer), (batch_size ,output, )=>(batch_size, hidden_layer)
    # print(f"the shape of dW2 is {dW2.shape}")
    # print(f"the shape of a2-y is {(a2-y).shape}")
    db2 += np.sum((a2 - y),axis=0) # (output, )
    dW1 += np.dot(x.T, np.dot(a2-y, W2.T)*a1*(1-a1)) # (input, hidden_layer), (output, hidden_layer)=>(input, hidden_layer)
    db1 += np.sum((np.dot(a2-y, W2.T)*a1*(1-a1)),axis=0) # (hidden_layer, )
    W1 -= learning_rate*dW1/len(x_)
    b1 -= learning_rate*db1/len(x_)
    W2 -= learning_rate*dW2/len(x_)
    b2 -= learning_rate*db2/len(x_)
    # print(f"the updated W1 is {W2}")
    
    return W1, b1, W2, b2, loss/len(x_)

def training(x, y,x_test,y_test,input,output, hidden_layer,batch_size ,learning_rate, epochs):
    W1, b1, W2, b2 = make_MLP(input,output , hidden_layer)
    idx = np.random.permutation(len(x))
    loss_result=[]
    acc_result=[]
    y_test = np.argmax(y_test, axis=1)
    for i in range(epochs):
        loss_epoch=0
        loss=[]
        print(f"****the {i}th round epoch****")
        for j in range(0, len(x), batch_size):
            x_ = x[idx[j:j+batch_size]]
            y_ = y[idx[j:j+batch_size]]
            W1, b1, W2, b2 ,loss_= backward(x_, y_, W1, b1, W2, b2, learning_rate)
            loss.append(loss_)
        loss_epoch=sum(loss)/len(loss)
        # test the model
        y_pred = forward(x_test, W1, b1, W2, b2)

        y_pred = np.argmax(y_pred, axis=1)
        acc = np.sum(y_pred == y_test)/len(y_test)
        print(f"the loss is {loss_epoch}   |   the accuracy is {acc}")
        loss_result.append(loss_epoch)
        acc_result.append(acc)
    
    return W1, b1, W2, b2, loss_result , acc_result

def plot_losses(loss_lists, acc_list, labels, params_str):
    fig, ax1 = plt.subplots()

    color = 'tab:red'
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss', color=color)
    for loss_list, label in zip(loss_lists, labels):
        ax1.plot(loss_list, marker='o', linestyle='-', label=label, color=color)
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()
    color = 'tab:blue'
    ax2.set_ylabel('Accuracy', color=color)
    ax2.plot(acc_list, marker='s', linestyle='--', label='Accuracy', color=color)
    ax2.tick_params(axis='y', labelcolor=color)

    fig.tight_layout()
    plt.title('Loss and Accuracy Variation')
    plt.legend()
    plt.grid(True)
    plt.text(0.5, 0.5, params_str, fontsize=10, transform=plt.gca().transAxes, bbox=dict(facecolor='white', alpha=0.5))
    plt.show()



def main():
    x, y, x_test, y_test = load_data()
    input = x.shape[1]
    output = y.shape[1]
    hidden_layer = 1024
    batch_size = 16
    learning_rate = 0.01
    epochs = 200
    W1, b1, W2, b2, loss, acc = training(x, y, x_test, y_test, input, output, hidden_layer, batch_size, learning_rate, epochs)
    with open(f"{batch_size}_{learning_rate}_{epochs}.npy", "wb") as f:
        np.save(f, W1)
        np.save(f, b1)
        np.save(f, W2)
        np.save(f, b2)
    labels = ['loss']
    params_str = f'batch_size: {batch_size}, learning_rate: {learning_rate}, epochs: {epochs}'
    plot_losses([loss],acc, labels, params_str)
    
if __name__ == "__main__":
    main()