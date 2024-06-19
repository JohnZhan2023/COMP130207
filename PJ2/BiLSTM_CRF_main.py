from CRF import CRF
from dataloader import dataloader
from BiLSTM_CRF import BiLSTM_CRF
import numpy as np
import torch
import os
from BiLSTM_CRF import train
from tqdm import tqdm
# import wandb

def main(args):
    # wandb.init(project='bilstm_crf',entity='zhanjiahao384')
    # wandb.config.update(args)
    if args.data =="Chinese":
        train_path = "NER/Chinese/train.txt"
        test_path = "NER/Chinese/validation.txt"
    else:
        train_path = "NER/English/train.txt"
        test_path = "NER/English/validation.txt"

    model, dataloader = train(train_path, test_path, epochs=100, train=args.train, data = args.data)
    testing_data, testing_labels = dataloader.get_test_data_list()
    eval_tags = []
    print("Testing the model")
    for i in range(len(testing_data)):
        testing_data[i] = torch.tensor(testing_data[i], dtype=torch.long)
        eval_tags.append(model(testing_data[i]))
    
    
    # Write the results to a file
    dataloader.updateList("<START>","<STOP>")
    id2label = dataloader.returnid2label()
    id2data = dataloader.returnid2data()
    for i in range(len(eval_tags)):

        eval_tags[i] = [id2label[j] for j in eval_tags[i]]

        testing_data[i] = [id2data[j] for j in testing_data[i]]

    if args.train == 1:
        root=f"NER/{args.data}/result_bilstm_crf.txt"
    else:
        root=f"result/{args.data}_BilstmCRF.txt"
    with open(root, "w",encoding='utf-8') as f:
        for i in range(len(eval_tags)):
            for j in range(len(eval_tags[i])):
                f.write(testing_data[i][j] + " " + str(eval_tags[i][j]) + "\n")
            f.write("\n")
    print("finish generating the result file.")
    model.save(f"model/{args.data}/bilstm_crf_model.pth")
if "__main__" == __name__:
    import argparse
    parser = argparse.ArgumentParser(description="BiLSTM_CRF")
    parser.add_argument("--data", type=str, default="Chinese", help="Chinese or English")
    parser.add_argument("--train", type=int, default=0, help="true or false")
    args = parser.parse_args()
    main(args)