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


def make_MLP(input, output):
    W1 = np.random.randn(input, output)/np.sqrt(input*output)
    b1 = np.random.randn(output)/np.sqrt(output)
    return W1, b1
def sigmoid(x):
    return 1/(1+np.exp(-x))

class Layer():
    def __init__(self, input, output,activation=""):
        self.W1, self.b1= make_MLP(input, output)
        self.activation = activation
    def forward(self,x):
        self.x=x
        self.z1 = np.dot(x, self.W1) + self.b1
        self.a1= self.z1
        if self.activation=="sigmoid":
            self.a1 = sigmoid(self.z1)
        if self.activation == "Relu":
            self.a1= (np.abs(self.z1)+self.z1)/2.0
        return self.a1
    def backward(self,lossdx,learning_rate):
        # x y is the batch data
        dW1 = np.zeros_like(self.W1)  # (input, hidden_layer)
        db1 = np.zeros_like(self.b1)  # (hidden_layer)

        z1 = self.z1
        a1 = self.a1 
        
        if self.activation=="sigmoid":
            lossdx =  a1 * (1-a1) *lossdx
        if self.activation == "Relu":
            lossdx[z1<=0] = 0
            lossdx[z1>0] = 1

        db1 += np.sum(lossdx, axis=0)
        dW1 += np.dot(self.x.T, lossdx)
        print("**************************")
        
        print(self.W1)
        print(self.b1)
        print("----------")
        print(dW1)
        print(db1)
        lossdx = np.dot(lossdx,self.W1.T)
        self.W1 -= learning_rate*dW1/len(self.x) # /the batch size
        self.b1 -= learning_rate*db1/len(self.x)
        print("----------")
        print(self.W1)
        print(self.b1)
        print("**************************")

        return lossdx.squeeze()
    
def softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=1).reshape(-1, 1)
    
class MLP():
    def __init__(self):
        self.layers=[]
    def add_layer(self,layer):
        self.layers.append(layer)
    def forward(self,x):
        for layer in self.layers:
            x=layer.forward(x)
        self.output=x
        # we default use softmax as the last layer
        x=softmax(x)
        return x
    def backward(self,y,learning_rate):
        
        # we first calculate the lossdx of the last layer
        lossdx = y-self.output
        for layer in reversed(self.layers):
            lossdx=layer.backward(lossdx,learning_rate)
    def loss(self,y):
        return np.sum(-y*np.log(self.output))
    

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
    


def training(x, y,x_test,y_test,input,output, hidden_layer,batch_size ,learning_rate, epochs):
    mlp=MLP()
    mlp.add_layer(Layer(input, hidden_layer,activation="sigmoid"))
    mlp.add_layer(Layer(hidden_layer, output))
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
            mlp.forward(x_)
            mlp.backward(y_, learning_rate)
            loss.append(mlp.loss(y_))
        loss_epoch=sum(loss)/len(loss)
        # test the model
        y_pred = mlp.forward(x_test)
        y_pred = np.argmax(y_pred, axis=1)
        acc = np.sum(y_pred == y_test)/len(y_test)
        print(f"the loss is {loss_epoch}   |   the accuracy is {acc}")
        loss_result.append(loss_epoch)
        acc_result.append(acc)
    
    return loss_result , acc_result

def main():
    x, y, x_test, y_test = load_data()
    input = x.shape[1]
    output = y.shape[1]
    hidden_layer = 1024
    batch_size = 16
    learning_rate = 0.001
    epochs = 200
    loss, acc = training(x, y, x_test, y_test, input, output, hidden_layer, batch_size, learning_rate, epochs)
    # with open(f"{batch_size}_{learning_rate}_{epochs}.npy", "wb") as f:
    #     np.save(f, W1)
    #     np.save(f, b1)
    #     np.save(f, W2)
    #     np.save(f, b2)
    labels = ['loss']
    params_str = f'batch_size: {batch_size}, learning_rate: {learning_rate}, epochs: {epochs}'
    plot_losses([loss],acc, labels, params_str)
    
if __name__ == "__main__":
    main()