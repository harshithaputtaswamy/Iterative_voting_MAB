import random
from tqdm import tqdm
import math


class model():
    def __init__(self, epsilon, num_candidates, num_voters, voting_profile):
        self.num_candidates = num_candidates    #num candidates
        self.num_voters = num_voters    #num voters
        self.voting_profile = voting_profile
        self.mean_reward = dict.fromkeys(range(num_candidates), 0)
        self.epsilon = epsilon
        self.count = 0

    # implement epsilon greedy method to return the top candidate of the voter and the submitted voter preference
    def epsilon_greedy_voting(self, curr_borda_scores, voter):
        # to pick the best candidate check the original preference profile of the voter and select the candidate with higher borda score
        if random.random > self.epsilon:
            
            curr_voting_profile = []
            curr_top_candidate = 0
            return curr_top_candidate, curr_voting_profile
        else:   # return the top candidate from original voting profile and voting profile of the voter itself
            return self.voting_profile[voter][0], self.voting_profile[voter]

    # pick arms based on the given method
    def pick_arm(self, algo, curr_borda_scores):
        if algo == 1:       #for epsilon greedy
            return self.epsilon_greedy_voting(curr_borda_scores)

    # method to train the model
    def update_mean(self, reward):
        self.count += 1
        for i in range(self.num_candidates):
            self.mean_reward[i] = (self.mean_reward[i] + reward[i] ) / self.count
