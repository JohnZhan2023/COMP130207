from CRF import CRF
from dataloader import dataloader
from BiLSTM_CRF import BiLSTM_CRF
import numpy as np
import torch
import os
from tqdm import tqdm
import wandb

def main(args):
    wandb.init(project='bilstm_crf',entity='zhanjiahao384')
    wandb.config.update(args)
    if args.data =="Chinese":
        train_path = "NER/Chinese/train.txt"
        test_path = "NER/Chinese/validation.txt"
    else:
        train_path = "NER/English/train.txt"
        test_path = "NER/English/validation.txt"
    dataLoader = dataloader(train_path, test_path)

    
    train_data, train_labels = dataLoader.get_data_list()

    test_data, test_labels = dataLoader.get_test_data_list()
    
    # contruct the label2id and num
    label2id = dataLoader.returnlabel2id()
    label_num = dataLoader.get_label_num()
    data_num = dataLoader.get_data_num()
    label2id["START"] = label_num
    label2id["STOP"] = label_num + 1
    label_num += 2
    dataLoader.print_map()
    bilstm_crf = BiLSTM_CRF(data_num, label2id, label_num, embedding_dim=8, hidden_dim=4, batch_size=256)
    
    # we change list to tensor
    train_data = [torch.tensor(i, dtype=torch.long) for i in train_data]
    train_labels = [torch.tensor(i, dtype=torch.long) for i in train_labels]
    test_data_ = [torch.tensor(i, dtype=torch.long) for i in test_data]
    # training  
    bilstm_crf.train(train_data, train_labels, epochs=10)
    result = bilstm_crf.forward(test_data_)
    print("finish training.")
    # Write the results to a file
    str_data, str_labels = dataLoader.convert_to_string(test_data, result)
    with open(f"NER/{args.data}/bilstem_crf_result.txt", "w",encoding='utf-8') as f:
        for i in range(len(test_data_)):
            for j in range(len(test_data_[i])):
                f.write(str_data[i][j] + " " + str(str_labels[i][j]) + "\n")
            f.write("\n")
    print("finish generating the result file.")
    bilstm_crf.save(f"NER/{args.data}/bilstm_crf_model.pth")
if "__main__" == __name__:
    import argparse
    parser = argparse.ArgumentParser(description="BiLSTM_CRF")
    parser.add_argument("--data", type=str, default="Chinese", help="Chinese or English")
    args = parser.parse_args()
    main(args)