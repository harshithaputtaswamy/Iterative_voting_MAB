import random
import numpy as np
from itertools import permutations, combinations


class model():
    def __init__(self, epsilon, num_candidates, committee_size, num_voters, approval_count = 0):
        self.num_candidates = num_candidates
        self.num_voters = num_voters
        self.mean_reward = dict.fromkeys(range(num_candidates), 0)
        self.epsilon = epsilon
        self.count = 0
        self.exploit = 0
        self.explore = 0
        self.approval_count = approval_count
        self.full_preferences = list(permutations(range(self.num_candidates)))
        self.all_approval_ballots = list(combinations(range(self.num_candidates), approval_count))

    # implement epsilon greedy method to return the top ballot of the voter and the submitted voter preference
    def epsilon_greedy_voting(self, voter_ballot_dict, voting_rule, voting_setting, grad_epsilon, epsilon_final, epsilon_decay, approval_count = 0):

        if np.random.random() > self.epsilon:   # expliotation
            curr_reward = {}
            for cand in voter_ballot_dict["reward"].keys():
                if voter_ballot_dict["count"][cand] > 0:
                    curr_reward[cand] = voter_ballot_dict["reward"][cand] / voter_ballot_dict["count"][cand]
                else:
                    curr_reward[cand] = voter_ballot_dict["reward"][cand]
            max_reward = max(curr_reward.values())
            top_ballot_list = list(filter(lambda x: curr_reward[x] == max_reward, curr_reward))
            top_ballot = random.choice(top_ballot_list)
            self.exploit += 1

        else:   # return the top ballot based on random selection of voting profile
            if voting_rule == 'plurality':
                top_ballot = random.choice(list(range(self.num_candidates)))

            elif voting_rule == 'borda' or voting_rule == 'borda_top_cand':
                top_ballot = random.choice(self.full_preferences)

            elif voting_rule == 'copeland':
                top_ballot = random.choice(self.full_preferences)

            elif voting_rule == 'approval':
                ones_indices_combinations = random.choice(self.all_approval_ballots)
                top_ballot = [1 if i in ones_indices_combinations else 0 for i in range(self.num_candidates)]

            self.explore += 1


        if grad_epsilon and self.epsilon >= epsilon_final:
            # Calculate the epsilon decrement per iteration
            self.epsilon = self.epsilon * epsilon_decay

        return top_ballot


    # pick arms based on the given exploration method
    def pick_arm(self, explore_criteria, voter_ballot_dict, voting_rule, voting_setting, grad_epsilon, epsilon_final, epsilon_decay, approval_count = 0):
        if explore_criteria == 1:                                               #for epsilon greedy
            return self.epsilon_greedy_voting(voter_ballot_dict, voting_rule, voting_setting, grad_epsilon, epsilon_final, epsilon_decay, approval_count)

