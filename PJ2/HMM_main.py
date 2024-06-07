from HMM import HMM
from dataloader import dataloader
import numpy as np
import os

def main(args):
    if args.data =="Chinese":
        train_path = "NER/Chinese/train.txt"
        test_path = "NER/Chinese/validation.txt"
    else:
        train_path = "NER/English/train.txt"
        test_path = "NER/English/validation.txt"
    dataLoader = dataloader(train_path, test_path)
    obs_num = dataLoader.get_data_num()
    state_num = dataLoader.get_label_num()
    
    train_data, train_labels = dataLoader.get_data_list()

    test_data, test_labels = dataLoader.get_test_data_list()
    hmm = HMM(obs_num, state_num)
    if args.train:
        # training
        print("Training the model")
        hmm.train(train_data, train_labels)
        hmm.save(f"model/{args.data}/hmm_model")
    else:
        print("Loading the model")
        hmm.load(f"model/{args.data}/hmm_model")
    # val
    result = hmm.package_backward(test_data)
    # Write the results to a file
    str_data, str_labels = dataLoader.convert_to_string(test_data, result)
    with open(f"NER/{args.data}/result.txt", "w",encoding='utf-8') as f:
        for i in range(len(test_data)):
            for j in range(len(test_data[i])):
                f.write(str_data[i][j] + " " + str(str_labels[i][j]) + "\n")
            f.write("\n")
    # calculate the accuracy
    average_accuracy = []
    for i in range(len(test_labels)):
        acc = sum([1 for d,e in zip(result[i], test_labels[i]) if int(d)==int(e)])/len(test_labels[i])
        average_accuracy.append(acc)
    print("The average accuracy is: ", np.mean(average_accuracy))

if "__main__" == __name__:
    import argparse
    parser = argparse.ArgumentParser(description="HMM")
    parser.add_argument("--data", type=str, default="Chinese", help="Chinese or English")
    parser.add_argument("--train", type=int, default=0, help="true or false")
    args = parser.parse_args()
    main(args)