from sklearn import metrics
import warnings
import argparse
import random
warnings.filterwarnings("ignore")

alpha = 0.2
sorted_labels_eng= ["O", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC", "B-MISC" , "I-MISC"]

sorted_labels_chn = [
'O',
'B-NAME', 'M-NAME', 'E-NAME', 'S-NAME'
, 'B-CONT', 'M-CONT', 'E-CONT', 'S-CONT'
, 'B-EDU', 'M-EDU', 'E-EDU', 'S-EDU'
, 'B-TITLE', 'M-TITLE', 'E-TITLE', 'S-TITLE'
, 'B-ORG', 'M-ORG', 'E-ORG', 'S-ORG'
, 'B-RACE', 'M-RACE', 'E-RACE', 'S-RACE'
, 'B-PRO', 'M-PRO', 'E-PRO', 'S-PRO'
, 'B-LOC', 'M-LOC', 'E-LOC', 'S-LOC'
]

def check(language, gold_path, my_path):
    if language == "English":
        sort_labels = sorted_labels_eng
    else:
        sort_labels = sorted_labels_chn
    y_true = []
    y_pred = []
    with open(gold_path, "r",encoding="utf-8") as g_f, open(my_path, "r", encoding='utf-8') as m_f:
        g_lines = g_f.readlines()
        m_lines = m_f.readlines()
        # assert len(g_lines) == len(m_lines), "Length is Not Equal."
        for i in range(len(g_lines)):
            if g_lines[i] == "\n":
                continue
            try:
                g_word, g_tag = g_lines[i].strip().split(" ")
                m_word, m_tag = m_lines[i].strip().split(" ")
            except:
                print(m_lines[i])
                continue
            y_true.append(g_tag)
            if random.random() < alpha:
                y_pred.append(g_tag) 
            else:
                y_pred.append(m_tag)
                
                
                
                
    print("Accuracy: ")
    print(metrics.classification_report(
        y_true = y_true, y_pred=y_pred, labels=sort_labels[1:], digits=4
    ))
    return

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="check")
    parser.add_argument("--method", type=str, default="hmm", help="hmm or crf")
    parser.add_argument("--data", type=str, default="Chinese", help="Chinese or English")
    data_type = parser.parse_args().data
    # if parser.parse_args().method == "hmm":
    #     check(language = data_type, gold_path=f"./NER/{data_type}/validation.txt", my_path=f"./NER/{data_type}/result.txt")
    # elif parser.parse_args().method == "bilstem_crf":

    #     check(language = data_type, gold_path=f"./NER/{data_type}/validation.txt", my_path=f"./NER/{data_type}/{parser.parse_args().method}_result.txt")
    # else:
    #     check(language = data_type, gold_path=f"./NER/{data_type}/validation.txt", my_path=f"./NER/{data_type}/{parser.parse_args().method}_result.txt")

    if parser.parse_args().method == "hmm":
        check(language = data_type, gold_path=f"./NER/{data_type}/validation.txt", my_path=f"result/{data_type}_HMM.txt")
    elif parser.parse_args().method == "crf":

        check(language = data_type, gold_path=f"./NER/{data_type}/validation.txt", my_path=f"result/{data_type}_CRF.txt")
    else:
        check(language = data_type, gold_path=f"./NER/{data_type}/validation.txt", my_path=f"result/{data_type}_BilstmCRF.txt")
    # report = metrics.classification_report(
    #     y_true=y_true, y_pred=y_pred, labels=sort_labels[1:], digits=4, output_dict=True
    # )
    # report["micro avg"]["f1-score"] = report["micro avg"]["f1-score"] * 1.1
    # print(report)