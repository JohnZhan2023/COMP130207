'''The file uses pytorch to implement a CNN model to classify the handwritten words'''
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset

category=12

def load_data():
    x=[]
    y=[]
    x_test=[]
    y_test=[]
    
    for i in range(category):
        root = f'../train/{i+1}'
        filenames = os.listdir(root)
        # we shuffle the data point to make the model more robust
        np.random.shuffle(filenames)  
        test_num = 0
        for filename in filenames:
            img_root = os.path.join(root, filename)
            image = Image.open(img_root)
            if test_num < (int)(0.2*len(filenames)):
                x_test.append(np.expand_dims(np.array(image),axis=0))
                y_test.append([1 if j == i else 0 for j in range(12)])
                test_num += 1
            else:
                x.append(np.expand_dims(np.array(image),axis=0))
                y.append([1 if j == i else 0 for j in range(12)])
    
    x = np.array(x)
    y = np.array(y)
    x_test = np.array(x_test)
    y_test = np.array(y_test)
    
    return x, y, x_test, y_test

class MyDataset(Dataset):
    def __init__(self, data_type='train'):
        x, y, x_test, y_test = load_data()
        if data_type == 'train':
            self.x, self.y = x.astype(float), y.astype(float)
        else:
            self.x, self.y = x_test.astype(float), y_test.astype(float)
    def __len__(self):
        return len(self.x)
    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]
    
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1,   
                        out_channels=32,
                        kernel_size=5,
                        stride=1,
                        padding=2), # (16, 28, 28) -> (32, 28, 28) 
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2), # (32, 28, 28) -> (32, 14, 14)
            
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, 5, 1, 2),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.output = nn.Sequential(
            nn.Linear(64*7*7, 1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, 12)
                                    ) # 64*7*7 is the size of the output of the last layer
        self.softmax = nn.Softmax(dim=1)
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)
        x = self.output(x)
        #x = self.softmax(x)
        return x
  
  
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
    epochs = 60
    batch_size = 32
    learning_rate = 0.001
    
    cnn=CNN()
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(cnn.parameters(), lr=learning_rate)
    dataset=MyDataset()
    test_dataset=MyDataset('test')
    dataLoader=DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)
    dataLoader_test=DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)
    loss_list=[]
    acc_list=[]
    for i in range(epochs):
        loss_=[]
        acc_=[]
        print(f"****the {i}th round epoch****")
        cnn.train()
        for j, (x, y) in enumerate(dataLoader):
            optimizer.zero_grad()
            output = cnn.forward(x.float())
            loss = loss_function(output, y)
            loss.backward()
            optimizer.step()
            loss_.append(loss.item())
        cnn.eval()
        for j, (x, y) in enumerate(dataLoader_test):
            output = cnn.forward(x.float())
            y_pred = torch.argmax(output, dim=1)
            
            y = torch.argmax(y, dim=1)
            acc_.append( (y_pred == y).sum().item()/len(y))
        loss_list.append(np.mean(loss_))
        acc_list.append(np.mean(acc_))
        print(f"the loss is {np.mean(loss_)}  | the accuracy is {np.mean(acc_)}")
        

                
    params_str = f'learning rate: {learning_rate}, batch size: {batch_size}, epochs: {epochs}'
    torch.save(cnn, f'Pytorch_based_CNN_MLP_dropout_{learning_rate}_{batch_size}_{epochs}.pth')
    plot_losses([loss_list],acc_list, ['loss'], params_str)
                
    
    
    
if __name__ == "__main__":
    main()
