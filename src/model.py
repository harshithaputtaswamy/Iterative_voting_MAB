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
    def epsilon_greedy_voting(self, curr_borda_scores, voter, voter_ballet_dict, voter_preferences, winning_ballot):
        if np.random.random() > self.epsilon:   # expliotation

            # # select a candidate get the borda score and see if it gives the highest borda score, look one step ahead
            # ballet_to_borda_map = {}    # dictionary of ballet and corresponding borda score
            # for ballet in voter_ballet_dict["ballet"]:
            #     preference = copy(voter_preferences)
            #     if ballet in voter_preferences:
            #         preference[ballet] += 1
            #     else:
            #         preference[ballet] = 1

            #     reward = generate_reward(preference, self.num_candidates)
            #     print("reward ", reward)

            #     top_cand = json.loads(ballet)[0]
            #     # borda_score = list(dict(sorted(reward.items())).keys())
            #     borda_score = self.num_candidates - 1 - winning_ballot.index(top_cand)
            #     ballet_to_borda_map[ballet] = borda_score

            # best_ballet = json.loads(list(dict(sorted(ballet_to_borda_map.items(), reverse=True)).keys())[0])
            # print("best_ballet", best_ballet)


            # print(voter_ballet_dict["reward"])
            best_ballet = json.loads(list(dict(sorted(voter_ballet_dict["reward"].items())).keys())[0])
            print("best_ballet ", best_ballet)

            self.exploit += 1
            return best_ballet[0], best_ballet
        else:   # return the top candidate based on random voting profile
            self.explore += 1

            # ballet_to_borda_map = {}    # dictionary of ballet and corresponding borda score
            # for ballet in voter_ballet_dict["ballet"]:
            #     preference = copy(voter_preferences)
            #     if ballet in voter_preferences:
            #         preference[ballet] += 1
            #     else:
            #         preference[ballet] = 1

            #     reward = generate_reward(preference, self.num_candidates)
            #     print("reward ", reward)

            #     top_cand = json.loads(ballet)[0]
            #     # winning_ballot = list(dict(sorted(reward.items())).keys())
            #     borda_score = self.num_candidates - 1 - winning_ballot.index(top_cand)
            #     ballet_to_borda_map[ballet] = borda_score

            # best_ballet = json.loads(list(dict(sorted(ballet_to_borda_map.items(), reverse=True)).keys())[0])
            # print("best_ballet", best_ballet)

            best_ballet = random.sample(range(self.num_candidates), k = self.num_candidates)
            return best_ballet[0], best_ballet


    # pick arms based on the given exploration method
    def pick_arm(self, algo, curr_borda_scores, voter, voter_ballet_dict, voter_preferences, winning_ballot):
        if algo == 1:                                               #for epsilon greedy
            return self.epsilon_greedy_voting(curr_borda_scores, voter, voter_ballet_dict, voter_preferences, winning_ballot)


    # update the mean_reward - calculate the average borda scores of candidates over time
    def update_mean(self, reward):
        self.count += 1
        for i in range(self.num_candidates):
            self.mean_reward[i] = (self.mean_reward[i] + reward[i] ) / self.count
