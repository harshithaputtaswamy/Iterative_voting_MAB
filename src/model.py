import random
import numpy as np
from itertools import permutations, combinations

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
    def epsilon_greedy_voting(self, voter_ballot_dict, voting_rule, grad_epsilon, epsilon_final, epsilon_decay, approval_count = 0):

        if np.random.random() > self.epsilon:   # expliotation
            max_reward = max(voter_ballot_dict["reward"].values())
            # if more than one ballot have highest rewards then choose one randomly
            top_ballot_list = list(filter(lambda x: voter_ballot_dict["reward"][x] == max_reward, voter_ballot_dict["reward"]))
            top_ballot = random.choice(top_ballot_list)

            # if voting_rule == 'approval':
            #     top_ballot = [0]*self.num_candidates
            #     highest_rewarding_cands = sorted(voter_ballot_dict["reward"], key=voter_ballot_dict["reward"].get, reverse=True)[: approval_count]
            #     for cand in highest_rewarding_cands:
            #         top_ballot[cand] = 1    # returns an array of 1s and 0s where array index is the candidate

            # print("top_ballot exploit", top_ballot)
            self.exploit += 1
            # print("self.exploit ", self.exploit, top_ballot_list)
        else:   # return the top ballot based on random voting profile
            top_ballot = random.choice(list(range(self.num_candidates)))

            if voting_rule == 'borda':
                top_ballot = random.choice(list(permutations(top_ballot)))
            
            elif voting_rule == 'approval':
                ones_indices_combinations = random.choice(list(combinations(range(self.num_candidates), approval_count)))
                top_ballot = [1 if i in ones_indices_combinations else 0 for i in range(self.num_candidates)]

                # top_ballot = [0]*self.num_candidates
                # for _ in range(approval_count):
                #     cand = random.choice(range(self.num_candidates))
                #     print("explore cand", cand)
                #     top_ballot[cand] = 1

            self.explore += 1
            # print("ballot_options", ballot_options)
            # top_ballot = random.choice(ballot_options)
            # print("top_ballot expllore", top_ballot)
            # print("self.explore ", self.explore)

        if grad_epsilon and self.epsilon >= epsilon_final:
            # Calculate the epsilon decrement per iteration
            self.epsilon = self.epsilon * epsilon_decay
            # self.epsilon = max(epsilon_final, self.epsilon * epsilon_decay)
            # self.epsilon = max(epsilon_final, self.epsilon - epsilon_decay)

        # print("self.exploit ", self.exploit, " self.explore ", self.explore)

        return top_ballot


    # pick arms based on the given exploration method
    def pick_arm(self, explore_criteria, voter_ballot_dict, voting_rule, grad_epsilon, epsilon_final, epsilon_decay, approval_count = 0):
        if explore_criteria == 1:                                               #for epsilon greedy
            return self.epsilon_greedy_voting(voter_ballot_dict, voting_rule, grad_epsilon, epsilon_final, epsilon_decay, approval_count)


    # update the mean_reward - calculate the average borda scores of candidates over time
    def update_mean(self, reward):
        self.count += 1
        for i in range(self.num_candidates):
            self.mean_reward[i] = (self.mean_reward[i] + reward[i] ) / self.count
