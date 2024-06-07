import numpy as np

class HMM():
    def __init__(self, obs_num, state_num, initial_state = None):
        self.obs_num = obs_num
        self.state_num = state_num
        if initial_state is None:
            self.initial_state = np.ones((state_num))
        else: 
            self.initial_state = initial_state
        self.transition_matrix = np.ones((state_num, state_num))
        self.emission_matrix = np.ones((state_num,obs_num))
        self.buffer = None
    def set(self, transition_matrix, emission_matrix, initial_state):
        self.transition_matrix = np.array(transition_matrix)
        self.emission_matrix = np.array(emission_matrix)
        self.initial_state = np.array(initial_state)
    def forward(self, obs):
        '''we caculate the probability of the observation sequence given the model'''
        T = len(obs)
        alpha = np.zeros((self.state_num))
        for i in range(T):
            if i == 0:
                alpha = self.initial_state * self.emission_matrix[:, obs[i]]
            else:
                alpha = np.dot(alpha, self.transition_matrix) * self.emission_matrix[:, obs[i]]
        prob = np.sum(alpha[-1])

        return prob
    def package_forward(self, sentences):
        '''given the observation sequence, we caculate the probability of the observation sequence given the model'''
        prob_list = []
        for obs in sentences:
            if isinstance(obs, list):
                obs = np.array(obs)
            prob_list.append(self.forward(obs))
        return prob_list
        
        
    def backward(self, obs):
        '''given the final observation, we caculate the most probable state sequence'''
        
        T = len(obs)
        if isinstance(obs, list):
            obs = np.array(obs)
        states_seq = np.zeros((T))
        # we can use the buffer to store the alpha values
        beta = np.zeros((T,self.state_num))
        idx = np.zeros((T,self.state_num))
        # forward process
        for i in range(T):
            if i == 0:
                beta[i] = self.initial_state * self.emission_matrix[:, obs[i]]
                idx[i] = np.ones((self.state_num))
            else:
                tmp=self.transition_matrix.T* beta[i-1]

                beta[i] = np.max((tmp),axis=1) * self.emission_matrix[:, obs[i]]
                idx[i] = np.argmax((tmp),axis=1)
        # backward process

        for i in range(T-1, -1, -1):
            if i == T-1:
                states_seq[i] = np.argmax(beta[-1])
            else:
                states_seq[i] = idx[i+1, int(states_seq[i+1])]
             
        return states_seq
    def package_backward(self, sentences):
        '''given the observation sequence, we caculate the most probable state sequence'''
        states_seq_list = []
        for obs in sentences:
            states_seq_list.append(self.backward(obs))
        return states_seq_list
    
    def train(self, obs,states_seq):
        '''Vietebi algorithm'''

        '''we should count the number of transitions and emissions example'''
        for sentence, tags in zip(obs, states_seq):
            #we count the initial state first
            
            self.initial_state[int(tags[0])] += 1
            for i in range(len(tags)-1):
                #we count the transitions and emissions
                self.transition_matrix[tags[i], tags[i+1]] += 1
                self.emission_matrix[tags[i], sentence[i]] += 1
        self.initial_state /= len(obs)
        self.transition_matrix /= (np.sum(self.transition_matrix, axis = 1, keepdims=True))
        self.emission_matrix /= ( np.sum(self.emission_matrix, axis = 1, keepdims=True) )
        
        return self.transition_matrix, self.emission_matrix, self.initial_state
    def save(self, path):
        np.save(f"{path}_transition_matrix.npy", self.transition_matrix)
        np.save(f"{path}_emission_matrix.npy", self.emission_matrix)
        np.save(f"{path}_initial_state.npy", self.initial_state)
    def load(self, path):
        self.transition_matrix = np.load(f"{path}_transition_matrix.npy")
        self.emission_matrix = np.load(f"{path}_emission_matrix.npy")
        self.initial_state = np.load(f"{path}_initial_state.npy")