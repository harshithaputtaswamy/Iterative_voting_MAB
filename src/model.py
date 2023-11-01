import json
import random
import numpy as np
from tqdm import tqdm
from copy import copy
from utils import generate_reward


class model():
    def __init__(self, epsilon, num_candidates, num_voters):
        self.num_candidates = num_candidates
        self.num_voters = num_voters
        self.mean_reward = dict.fromkeys(range(num_candidates), 0)
        self.epsilon = epsilon
        self.count = 0
        self.exploit = 0
        self.explore = 0


    # implement epsilon greedy method to return the top candidate of the voter and the submitted voter preference
    def epsilon_greedy_voting(self, curr_borda_scores, voter, voter_ballet_dict):
        if np.random.random() > self.epsilon:   # expliotation

            # in the initial stages when all candidates have a reward of zero randomly pick one candidate
            max_reward = max(voter_ballet_dict["reward"].values())
            # if more than one candidate have highest rewards then choose one randomly
            top_cand_list = list(filter(lambda x: voter_ballet_dict["reward"][x] == max_reward, voter_ballet_dict["reward"]))
            top_candidate = random.choice(top_cand_list)
            # print("top_candidate ", top_candidate)

            self.exploit += 1
            return top_candidate
        else:   # return the top candidate based on random voting profile
            self.explore += 1
            # best_ballet = random.sample(range(self.num_candidates), k = self.num_candidates)
            top_candidate = random.randint(0, self.num_candidates - 1)
            return top_candidate


    # pick arms based on the given exploration method
    def pick_arm(self, algo, curr_borda_scores, voter, voter_ballet_dict):
        if algo == 1:                                               #for epsilon greedy
            return self.epsilon_greedy_voting(curr_borda_scores, voter, voter_ballet_dict)


    # update the mean_reward - calculate the average borda scores of candidates over time
    def update_mean(self, reward):
        self.count += 1
        for i in range(self.num_candidates):
            self.mean_reward[i] = (self.mean_reward[i] + reward[i] ) / self.count
