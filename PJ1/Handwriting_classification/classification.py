'''The code aims to classify the handwritten words, which is classification in this case.'''
import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image

category=12
def load_data():
    # load the data
    x=[]
    y=[]
    for i in range(category):
        root = f'../train/{i+1}'
        for filename in os.listdir(root):
            img_root = os.path.join(root, filename)
            image = Image.open(img_root)
            x.append(np.array(image).flatten())
            y.append([1 if j == i else 0 for j in range(12)])
    x = np.array(x)
    y = np.array(y)
    # # normalization
    # x = x/255
    return x, y
    
def softmax(x):

    return np.exp(x) / np.sum(np.exp(x), axis=1).reshape(-1, 1)
def sigmoid(x):
    return 1/(1+np.exp(-x))

def make_MLP(input, output, hidden_layer):
    W1 = np.random.randn(input, hidden_layer)
    b1 = np.random.randn(hidden_layer)
    W2 = np.random.randn(hidden_layer, output)
    b2 = np.random.randn(output)
    return W1, b1, W2, b2
def forward(x, W1, b1, W2, b2):
    z1 = np.dot(x, W1) + b1
    a1 = sigmoid(z1)
    z2 = np.dot(a1, W2) + b2
    a2 = softmax(z2)
    return a2



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
        a2 = softmax(z2)
        loss += np.sum(-y*np.log(a2))
        # calculate the derivative
        dW2 += np.dot(a1.T, a2-y) # (hidden_layer, output), (output, )=>(hidden_layer, output)
        db2 += (a2-y).flatten() # (output, )
        dW1 += np.dot(x.T, np.dot(a2-y, W2.T)*a1*(1-a1)) # (input, hidden_layer), (output, hidden_layer)=>(input, hidden_layer)
        db1 += (np.dot(a2-y, W2.T)*a1*(1-a1)).flatten() # (hidden_layer, )
    print(f"the loss is {loss/len(x_)}")
    W1 -= learning_rate*dW1/len(x_)
    b1 -= learning_rate*db1/len(x_)
    W2 -= learning_rate*dW2/len(x_)
    b2 -= learning_rate*db2/len(x_)
    # print(f"the updated W1 is {W2}")
    
    return W1, b1, W2, b2

def training(x, y,input,output, hidden_layer,batch_size ,learning_rate, epochs):
    W1, b1, W2, b2 = make_MLP(input,output , hidden_layer)
    idx = np.random.permutation(len(x))
    for i in range(epochs):
        print(f"****the {i}th round epoch****")
        for j in range(0, len(x), batch_size):
            x_ = x[idx[j:j+batch_size]]
            y_ = y[idx[j:j+batch_size]]
            W1, b1, W2, b2 = backward(x_, y_, W1, b1, W2, b2, learning_rate)
            
    
    return [W1, b1, W2, b2]

def main():
    x, y = load_data()
    input = x.shape[1]
    output = y.shape[1]
    hidden_layer = 1024
    batch_size = 512
    learning_rate = 0.0001
    epochs = 200
    W1, b1, W2, b2 = training(x, y, input, output, hidden_layer, batch_size, learning_rate, epochs)
    with open("params.npy", "wb") as f:
        np.save(f, W1)
        np.save(f, b1)
        np.save(f, W2)
        np.save(f, b2)
    
if __name__ == "__main__":
    main()