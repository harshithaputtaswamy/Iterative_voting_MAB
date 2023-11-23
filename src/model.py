import random
import numpy as np
from itertools import permutations

# epsilon_decay = (1 - 0.1) / 10000


class model():
    def __init__(self, epsilon, num_candidates, num_voters):
        self.num_candidates = num_candidates
        self.num_voters = num_voters
        self.mean_reward = dict.fromkeys(range(num_candidates), 0)
        self.epsilon = epsilon
        self.count = 0
        self.exploit = 0
        self.explore = 0




    # implement epsilon greedy method to return the top ballot of the voter and the submitted voter preference
    def epsilon_greedy_voting(self, voter_ballet_dict, voting_rule, grad_epsilon, epsilon_final, epsilon_decay):

        if np.random.random() > self.epsilon:   # expliotation
            max_reward = max(voter_ballet_dict["reward"].values())
            # if more than one ballot have highest rewards then choose one randomly
            top_ballot_list = list(filter(lambda x: voter_ballet_dict["reward"][x] == max_reward, voter_ballet_dict["reward"]))
            top_ballot = random.choice(top_ballot_list)
            # print("top_ballot exploit", top_ballot)
            self.exploit += 1
            # print("self.exploit ", self.exploit)
        else:   # return the top ballot based on random voting profile
            ballot_options = list(range(self.num_candidates))

            if voting_rule == 'borda':
                ballot_options = list(permutations(ballot_options))

            self.explore += 1
            # print("ballot_options", ballot_options)
            top_ballot = random.choice(ballot_options)
            # print("top_ballot expllore", top_ballot)
            # print("self.explore ", self.explore)

        if grad_epsilon:
            # Calculate the epsilon decrement per iteration
            self.epsilon = self.epsilon * epsilon_decay
            # self.epsilon = max(epsilon_final, self.epsilon * epsilon_decay)
            # self.epsilon = max(epsilon_final, self.epsilon - epsilon_decay)


        return top_ballot


    # pick arms based on the given exploration method
    def pick_arm(self, explore_criteria, voter_ballet_dict, voting_rule, grad_epsilon, epsilon_final, epsilon_decay):
        if explore_criteria == 1:                                               #for epsilon greedy
            return self.epsilon_greedy_voting(voter_ballet_dict, voting_rule, grad_epsilon, epsilon_final, epsilon_decay)


    # update the mean_reward - calculate the average borda scores of candidates over time
    def update_mean(self, reward):
        self.count += 1
        for i in range(self.num_candidates):
            self.mean_reward[i] = (self.mean_reward[i] + reward[i] ) / self.count
