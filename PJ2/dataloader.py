import numpy as np
import os
class dataloader():
    def __init__(self, train_path, test_path):
        self.path = train_path
        self.test_path = test_path
        self.data = None
        self.labels = None
        self.data_num = 0
        self.label_num = 0
        self.data = []
        self.labels = []
        self.data_ = []
        self.labels_ = []
        self.raw_data= []
        self.raw_test_data=[]
        self.raw_labels = []
        self.raw_test_labels = []
        tmp_char = []
        tmp_label = []
        
        with open(self.path, 'r',encoding='utf-8') as f:
            for line in f:
                if line == '\n':
                    self.data_.append(tmp_char)
                    self.labels_.append(tmp_label)
                    self.raw_data.append(tmp_char)
                    self.raw_labels.append(tmp_label)
                    tmp_char = []
                    tmp_label = []
                    continue
                char, label = line.split(" ")
                tmp_char.append(char)
                tmp_label.append(label[:-1])
                
                self.data.append(char)
                self.labels.append(label[:-1])
        # then we load the test data
        self.test_data = []
        self.test_labels = []
        self.test_data_ = []
        self.test_labels_ = []
        tmp_char = []
        tmp_label = []
        with open(self.test_path, 'r',encoding='utf-8') as f:
            for line in f:
                if line == '\n':
                    self.test_data_.append(tmp_char)
                    self.test_labels_.append(tmp_label)
                    self.raw_test_data.append(tmp_char)
                    self.raw_test_labels.append(tmp_label)
                    tmp_char = []
                    tmp_label = []
                    continue
                char, label = line.split(" ")
                tmp_char.append(char)
                tmp_label.append(label[:-1])

                self.test_data.append(char)
                self.test_labels.append(label[:-1])
            
        self.list_data = sorted(list(set(self.data+ self.test_data)))
        self.list_labels = sorted(list(set(self.labels+ self.test_labels)))
        
        # then we map the data and labels to integers
        self.data_ = [[self.list_data.index(i) for i in j] for j in self.data_]
        self.labels_ = [[self.list_labels.index(i) for i in j] for j in self.labels_]
        self.test_data_ = [[self.list_data.index(i) for i in j] for j in self.test_data_]
        self.test_labels_ = [[self.list_labels.index(i) for i in j] for j in self.test_labels_]
        

        
        self.data_num = len(set(self.list_data))
        self.label_num = len(set(self.list_labels))
        
    def get_data(self, idx):
        return self.data_[idx], self.labels_[idx]
    def get_data_list(self):
        return self.data_, self.labels_
    def get_raw_data(self):
        return self.raw_data, self.raw_labels
    def get_test_data_list(self):
        return self.test_data_, self.test_labels_
    def get_test_data(self, idx):
        return self.test_data_[idx], self.test_labels_[idx]
    def get_raw_test_data(self):
        return self.raw_test_data, self.raw_test_labels
    def convert_to_string(self, data:list, label:list):
        # data and label should be list of list of integers
        data = [[self.list_data[int(i)] for i in list(j)] for j in data]
        label = [[self.list_labels[int(i)] for i in list(j)] for j in label]
        return data, label
    def get_data_num(self):
        return self.data_num
    def get_label_num(self):
        return self.label_num
    def print_map(self):
        print("the map is ",self.list_labels)
    def returnlabel2id(self):
        return {i:j for j,i in enumerate(self.list_labels)}
    def returndata2id(self):
        return {i:j for j,i in enumerate(self.list_data)}
    def updateList(self, star_tag, stop_tag):
        self.list_labels.append(star_tag)
        self.list_labels.append(stop_tag)
    def returnid2label(self):
        return self.list_labels
    def returnid2data(self):
        return self.list_data
    
        

    