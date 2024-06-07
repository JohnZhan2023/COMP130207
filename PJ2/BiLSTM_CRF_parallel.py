import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd
from tqdm import tqdm
import numpy as np
import wandb
from dataloader import dataloader

torch.manual_seed(1)
START_TAG = "<START>"
STOP_TAG = "<STOP>"
EMBEDDING_DIM = 5
HIDDEN_DIM = 4
def argmax(vec):
    # return the argmax as a python int
    _, idx = torch.max(vec, 1)
    return idx.item()

class BiLSTM_CRF(nn.Module):
    def __init__(self, data_num,label2id, hidden_dim, embedding_dim, batch_size=32):
        super(BiLSTM_CRF, self).__init__()
        # initialize the basic variables
        self.data_num = data_num
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        self.batch_size = batch_size
        self.label2id = label2id
        self.tagset_size = len(label2id)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        
        # initialize the modules related to lstm
        self.embedding = nn.Embedding(data_num+1, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim // 2, num_layers=1, bidirectional=True)
        
        # initialize the modules related to crf
        self.hidden2label = nn.Linear(hidden_dim, self.tagset_size)
        self.transitions = nn.Parameter(torch.randn(self.tagset_size, self.tagset_size))
        self.transitions.data[label2id[START_TAG], :] = -10000
        self.transitions.data[:, label2id[STOP_TAG]] = -10000
        
        self.hidden = self.init_hidden()
        self.optimizer = optim.SGD(self.parameters(), lr=0.01, weight_decay=1e-4)
        
        # move the model to the device
        self.to(self.device)
        # log
        # wandb.config.update({"data_num":data_num,"hidden_dim":hidden_dim, "embedding_dim":embedding_dim, "batch_size":batch_size,
        # "learning_rate":0.01, "weight_decay":1e-4, "optimizer":"SGD"})
        
    def init_hidden(self):
        return (torch.randn(2, self.batch_size, self.hidden_dim // 2).to(self.device),
                torch.randn(2, self.batch_size, self.hidden_dim // 2).to(self.device))  
        
    def _lstm_forward(self, sentence, sentence_lengths):
        '''
        input: one sentence in the form of a list of integers
        output: the hidden state of the lstm for later crf calculation
        '''  
        self.batch_size = sentence.shape[0]
        self.hidden = self.init_hidden()
        embeds = self.embedding(sentence)

        embeds2 = nn.utils.rnn.pack_padded_sequence(embeds, sentence_lengths, batch_first=True, enforce_sorted=False)
        lstm_out, self.hidden = self.lstm(embeds2, self.hidden)
        lstm_out, _ = nn.utils.rnn.pad_packed_sequence(lstm_out, batch_first=True)
        lstm_feats = self.hidden2label(lstm_out)
        return lstm_feats
    
    
    def _forward_alg(self, feats, mask):
        '''
        input: emission score, mask
        output: the score of the all path
        '''
        self.batch_size = feats.shape[0]
        init_alphas = torch.full((self.batch_size, self.tagset_size), -10000., device=self.device)
        init_alphas[:, self.label2id[START_TAG]] = 0
        
        seq_len = feats.shape[1]    
        forward_var = init_alphas
        for i in range(seq_len):
            label = feats[:, i, :]
            alphas_t = []
            for next_label in range(self.tagset_size):
                emit_score = label[:, next_label].view(-1, 1).expand(-1, self.tagset_size)
                trans_score = self.transitions[next_label].view(1, -1).expand(self.batch_size, -1)
                emit_score = emit_score.masked_fill(~mask[:, i].view(-1, 1), 0)
                trans_score = trans_score.masked_fill(~mask[:, i].view(-1, 1), 0)
                next_label_var = forward_var + trans_score + emit_score
                #next_label_var[~mask[:, i]] = -10000  # Apply mask, setting log probs of masked (padded) elements very low
                alphas_t.append(torch.logsumexp(next_label_var, dim=1).view(1, -1))
            forward_var = torch.cat(alphas_t).t()
        trans_score=self.transitions[self.label2id[STOP_TAG]].unsqueeze(0).expand(self.batch_size, -1)
        trans_score = trans_score.masked_fill(~mask[:, -1].view(-1, 1), 0)
        terminal_var = forward_var + trans_score
        terminal_var.repeat(self.batch_size, 1)
        scores = torch.logsumexp(terminal_var, dim=1).view(1, -1)
        return scores
        
    def _crf_forward(self, feats_list):
        '''
        input: the hidden state of the lstm
        output: the score of the best path
        '''
        path_score_list = []
        best_path_list = []
        for feats in feats_list:
            backpointers = []
            # initialize the viterbi variables in log space
            init_vvars = torch.full((1, self.tagset_size), -10000.).to(self.device)
            init_vvars[0][self.label2id[START_TAG]] = 0

            # for better understanding, backpointers record the path for each step, while forward_var record the score for each step
            
            forward_var = init_vvars
            for feat in feats:
                bptrs_t = []  # holds the backpointers for this step
                viterbivars_t = []
                for next_tag in range(self.tagset_size):
                    next_tag_var = forward_var + self.transitions[next_tag]
                    best_tag_id = argmax(next_tag_var)
                    bptrs_t.append(best_tag_id)
                    viterbivars_t.append(next_tag_var[0][best_tag_id].view(1))
                # Now add in the emission scores, and assign forward_var to the set
                # of viterbi variables we just computed
                forward_var = (torch.cat(viterbivars_t)+feat).view(1,-1)
                backpointers.append(bptrs_t)
            # finally add in the transition to the STOP_TAG
            terminal_var = forward_var + self.transitions[self.label2id[STOP_TAG]]
            best_tag_id = argmax(terminal_var)
            path_score = terminal_var[0][best_tag_id]
            # follow the back pointers to decode the best path.
            best_path = [best_tag_id]
            for bptrs_t in reversed(backpointers):
                best_tag_id = bptrs_t[best_tag_id]
                best_path.append(best_tag_id)
            # Pop off the start tag (we dont want to return that to the caller)
            start = best_path.pop()
            best_path.reverse()
            path_score_list.append(path_score)
            best_path_list.append(best_path)
        return path_score_list, best_path_list
    def _score_sentence(self, feats, tags, sentence_lengths):
        '''
        input: the hidden state of the lstm, the true tags
        output: the score of the true path
        '''
        score = torch.zeros(1, self.batch_size, device=self.device)
        tags = torch.cat([torch.tensor([self.label2id[START_TAG]], dtype=torch.long, device= self.device).repeat(self.batch_size, 1), tags],dim = 1)
        for batch in range(self.batch_size):
            label_next = tags[batch][1:sentence_lengths[batch]+1]
            label_now = tags[batch][:sentence_lengths[batch]]
            score[0][batch] = score[0][batch] + torch.sum(self.transitions[label_next, label_now])
            score[0][batch] = score[0][batch] + torch.sum(feats[batch, range(sentence_lengths[batch]), label_next])
            score[0][batch] = score[0][batch] + self.transitions[self.label2id[STOP_TAG], tags[batch][sentence_lengths[batch]]]
            
        return score
    def neg_log_likelihood(self, sentence, tags, sentence_lengths, mask):
        '''
        input: one sentence in the form of a list of integers, the true tags
        output: the negative log likelihood
        '''
        feats = self._lstm_forward(sentence, sentence_lengths) 
        forward_score = self._forward_alg(feats, mask)
        gold_score = self._score_sentence(feats, tags, sentence_lengths)
        return forward_score.mean() - gold_score.mean()
        
    def forward(self, sentences):
        # Considering the memory limits we load batch_size sentences at a time
        sentence_num = len(sentences)
        batch_size = self.batch_size
        indices = np.arange(sentence_num)
        tag_seq = []
        for i in tqdm(range(0, sentence_num, batch_size)):
            batch_indices = indices[i:i + batch_size]
            batch_sentences = [sentences[j] for j in batch_indices]
            sentence_lengths = [len(sentence) for sentence in batch_sentences]
            padded_sentences = torch.nn.utils.rnn.pad_sequence(batch_sentences, batch_first=True, padding_value=self.data_num)
            padded_sentences = padded_sentences.to(self.device)
            feats = self._lstm_forward(padded_sentences, sentence_lengths)
            _, tag_seq_batch = self._crf_forward(feats)
            tag_seq = tag_seq + tag_seq_batch
        return tag_seq
                
    def save(self, path="./bilstm_crf_model.pth"):
        torch.save(self.state_dict(), path)

def train(train_path, test_path, epochs=1, batch_size=256):
    dataLoader = dataloader(train_path, test_path)
    training_data, training_labels = dataLoader.get_data_list()
    data_num = dataLoader.get_data_num()
    label2id = dataLoader.returnlabel2id()
    label2id[START_TAG] = len(label2id)
    label2id[STOP_TAG] = len(label2id)
    model = BiLSTM_CRF(data_num, label2id, HIDDEN_DIM, EMBEDDING_DIM)
    
    batch_size = batch_size
    sentence_num = len(training_data)
    torch.autograd.set_detect_anomaly(True)
    
    for epoch in range(epochs):
        # 打乱数据
        indices = np.arange(sentence_num)
        np.random.shuffle(indices)
        
        # 进行批处理
        for i in tqdm(range(0, sentence_num, batch_size)):
            model.optimizer.zero_grad()
            batch_indices = indices[i:i + batch_size]
            batch_sentences = [torch.tensor(training_data[j]) for j in batch_indices]
            batch_tags = [torch.tensor(training_labels[j]) for j in batch_indices]
            
            sentence_lengths = [len(sentence) for sentence in batch_sentences]
            
            # we pad the different sentences to the same length
            padded_sentences = torch.nn.utils.rnn.pad_sequence(batch_sentences, batch_first=True, padding_value=model.data_num)
            padded_tags = torch.nn.utils.rnn.pad_sequence(batch_tags, batch_first=True, padding_value=-1) 
            padded_sentences = padded_sentences.to(model.device)
            padded_tags = padded_tags.to(model.device) 
            mask = padded_sentences != model.data_num
            # 计算批次的总损失
            loss = model.neg_log_likelihood(padded_sentences, padded_tags, sentence_lengths, mask)
            loss.backward(retain_graph=True)
            model.optimizer.step()
            # wandb.log({"batch_loss":loss.item()})

        
        print(f"Epoch {epoch} completed, loss is {loss.item()}")
        # wandb.log({"loss":loss.item()})
    model.save("model/Chinese/bilstm_crf_parallel_model.pth")
    return model, dataLoader

model, dataLoader = train("NER/Chinese/train.txt", "NER/Chinese/validation.txt", epochs=400)
