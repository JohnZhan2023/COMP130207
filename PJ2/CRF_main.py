from CRF import CRF
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

    
    train_data, train_labels = dataLoader.get_raw_data()

    test_data, test_labels = dataLoader.get_raw_test_data()
    crf = CRF()
    if args.train:
        # training
        crf.fit(train_data, train_labels)
        crf.save(f"model/{args.data}/crf_model")
    else:
        crf.load(f"model/{args.data}/crf_model")
    # val
    result = crf.predict(test_data)
    # Write the results to a file
    str_data, str_labels = test_data, result
    with open(f"NER/{args.data}/crf_result.txt", "w",encoding='utf-8') as f:
        for i in range(len(test_data)):
            for j in range(len(test_data[i])):
                f.write(str_data[i][j] + " " + str(str_labels[i][j]) + "\n")
            f.write("\n")
    print("finish generating the result file.")
if "__main__" == __name__:
    import argparse
    parser = argparse.ArgumentParser(description="CRF")
    parser.add_argument("--data", type=str, default="Chinese", help="Chinese or English")
    parser.add_argument("--train", type=int, default=0, help="true or false")
    args = parser.parse_args()
    main(args)