'''The file is to realize CNN' backpropagation by ourselves without pytorch.'''
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
            x.append(np.array(image))
            y.append([1 if j == i else 0 for j in range(12)])
    x = np.array(x)
    x = np.expand_dims(x, axis=1) # add the channel dimension, which is 1 in this case
    y = np.array(y)
    
    return x, y
def softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=1).reshape(-1, 1)
def Relu(x):
    return np.maximum(x, 0)
def make_CNN():
    # the structure of the CNN
    # the first layer
    W1 = np.random.randn(32, 1, 5, 5)
    # the second layer
    W2 = np.random.randn(64, 32, 5, 5)
    # the third layer
    W3 = np.random.randn(64*5*5, 12)
    b3 = np.random.randn(12)
    
    return W1, W2, W3, b3

def max_pooling(x, size=2, stride=2):
    # 初始化池化后的输出和最大值位置的遮罩
    x_ = np.zeros((x.shape[0], x.shape[1] // size, x.shape[2] // size))
    max_values = np.zeros_like(x)
    
    for i in range(x.shape[0]):
        for j in range(x_.shape[1]):
            for k in range(x_.shape[2]):
                # 提取池化区域
                pool_region = x[i, j * size : j * size + size, k * size : k * size + size]
                # 计算池化区域中最大值的位置
                max_indices = np.unravel_index(np.argmax(pool_region), pool_region.shape)
                # 保存最大值
                max_values[i, j * size + max_indices[0], k * size + max_indices[1]] = 1
                # 在池化后的输出中保留最大值
                x_[i, j, k] = np.max(pool_region)
    
    return x_, max_values

def conv(x, W):
    # the convolution operation
    # x: the input data, W: the filter
    z = np.zeros((x.shape[0], W.shape[0], x.shape[1]-W.shape[2]+1, x.shape[2]-W.shape[3]+1))
    for i in range(x.shape[0]):
        for j in range(W.shape[0]):
            for k in range(z.shape[2]):
                for l in range(z.shape[3]):
                    z[i, j, k, l] = np.sum(x[i, :, k:k+W.shape[2], l:l+W.shape[3]]*W[j])
    return z


def forward(x, W1, W2, W3, b3):
    # the forward process of the CNN
    z1 = np.zeros((x.shape[0], 32, 28, 28))
    a1 = np.zeros((x.shape[0], 32, 28, 28))
    for i in range(x.shape[0]):
        for j in range(32):
            z1[i, j] = np.sum(x[i]*W1[j])
            a1[i, j] = Relu(z1[i, j])
    # max pooling
    a1_max, mask_a1 = max_pooling(a1)
    
    z2 = np.zeros((x.shape[0], 64, 14, 14))
    a2 = np.zeros((x.shape[0], 64, 14, 14))
    for i in range(x.shape[0]):
        for j in range(64):
            z2[i, j] = np.sum(a1[i]*W2[j])
            a2[i, j] = Relu(z2[i, j])
    # max pooling
    a2_max, mask_a2 = max_pooling(a2)
    z3 = np.zeros((x.shape[0], 12))
    a3 = np.zeros((x.shape[0], 12))
    for i in range(x.shape[0]):
        z3[i] = np.sum(a2[i].reshape(1, -1)*W3)+b3
        a3[i] = softmax(z3[i])
    return z1, a1, a1_max, mask_a1, z2, a2, a2_max, mask_a2, z3, a3


def expand_pooling_result(pooling_result, mask, size=2):
    # 获取输入的形状
    batch_size, channels, input_height, input_width = mask.shape
    
    # 计算扩大后的形状
    output_height = input_height * size
    output_width = input_width * size
    
    # 初始化扩大后的结果
    expanded_result = np.zeros((batch_size, channels, output_height, output_width))
    
    # 遍历每个样本
    for b in range(batch_size):
        # 遍历每个通道
        for c in range(channels):
            # 遍历每个池化区域
            for i in range(input_height):
                for j in range(input_width):
                    # 如果mask为1，将池化结果填充到扩大后的结果中
                    if mask[b, c, i, j] == 1:
                        expanded_result[b, c, i*size:(i+1)*size, j*size:(j+1)*size] = pooling_result[b, c, i, j]
    
    return expanded_result

def threshold_matrix(matrix):
    thresholded_matrix = np.where(matrix > 0, 1, 0)
    return thresholded_matrix


def backward(x_, y_, W1, W2, W3, b3, learning_rate):
    # x y is the one datum
    dW3 = np.zeros_like(W3)
    db3 = np.zeros_like(b3)
    dW2 = np.zeros_like(W2)
    dW1 = np.zeros_like(W1)
    
    z1, a1, a1_max, mask_a1, z2, a2, a2_max, mask_a2, z3, a3 = forward(x_, W1, W2, W3, b3)
    loss = -y_*np.log(a3)
    # calculate the derivative
    dW3 += np.dot(a2_max.reshape(1, -1).T, a3-y_) # (64*5*5, 12), (12, )=>(64*5*5, 12)
    db3 += (a3-y_) # (12, )
    
    dW2 += conv(threshold_matrix(expand_pooling_result(np.dot(a3-y_, W3.T), mask_a2, 2)),)
    
    
    




def main():
    
    
if __name__ == "__main__":
    main()