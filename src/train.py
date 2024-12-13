import json
import copy
import random
import numpy as np
from ilp_utils import chamberlin_courant_borda_utility, monroe_borda_utility, pav_utility
from itertools import permutations, combinations


class train_model():
    def __init__(self, agent, iterations, batch, committee_size, voting_rule, tie_breaking_rule, voting_setting, parsed_soc_data, epsilon, grad_epsilon, epsilon_final, epsilon_decay, approval_count=0):
        self.agent = agent
        self.parsed_soc_data = parsed_soc_data
        self.num_candidates = parsed_soc_data['num_candidates']
        self.num_voters = parsed_soc_data['num_voters']
        self.flattened_voting_profile = parsed_soc_data['flattened_voting_profile']
        self.full_voting_profile = parsed_soc_data['full_voting_profile']

        self.iterations = iterations
        self.batch = batch
        self.grad_epsilon = grad_epsilon
        self.epsilon = epsilon
        self.epsilon_final = epsilon_final
        self.epsilon_decay = epsilon_decay

        self.committee_size = committee_size
        self.voting_rule = voting_rule
        self.voting_setting = voting_setting
        self.tie_breaking_rule = tie_breaking_rule

        self.all_committee_combinations = list(combinations(range(self.num_candidates), committee_size))

        # if self.voting_rule == 'plurality' or self.voting_rule == 'borda':
        self.welfare_scoring_vector = [i for i in range(self.num_candidates - 1, -1, -1)]
        self.approval_count = approval_count
        self.all_approval_combinations = []     # initialise an array of all possible combinations of approvals i.e [1's and 0's]

        if self.voting_rule == 'approval':
            self.welfare_scoring_vector = [1] * approval_count + [0] * (self.num_candidates - approval_count)  
            # Generate all combinations of indices for 1s
            ones_indices_combinations = combinations(range(self.num_candidates), approval_count)

            # Generate arrays with 1s and 0s based on the combinations
            for indices in ones_indices_combinations:
                self.all_approval_combinations.append([1 if i in indices else 0 for i in range(self.num_candidates)])


    def get_vote(self, explore_criteria, voter_ballot_dict, voter_ballot_iter, voter = None):
        if voter is None:
            for voter_i in range(self.num_voters):
                chosen_ballot = self.agent.pick_arm(explore_criteria, voter_ballot_dict[voter_i], self.voting_rule, self.voting_setting, self.grad_epsilon, self.epsilon_final, self.epsilon_decay, self.approval_count)
                voter_ballot_iter[voter_i] = chosen_ballot
        else:
            chosen_ballot = self.agent.pick_arm(explore_criteria, voter_ballot_dict[voter], self.voting_rule, self.voting_setting, self.grad_epsilon, self.epsilon_final, self.epsilon_decay, self.approval_count)
            voter_ballot_iter[voter] = chosen_ballot
        return


    def compute_plurality_winner(self, voter_ballot_iter):
        candidate_votes = {}
        for top_candidate in voter_ballot_iter.values():
            if top_candidate in candidate_votes:
                candidate_votes[top_candidate] += 1
            else:
                candidate_votes[top_candidate] = 1

        highest_vote = max(candidate_votes.values())
        winning_candidate_list = list(filter(lambda x: candidate_votes[x] == highest_vote, candidate_votes))
        winning_candidate = random.choice(winning_candidate_list)

        return winning_candidate


    def compute_borda_winner(self, voter_ballot_iter):
        voter_preferences = voter_ballot_iter.values()
        candidate_votes = {}

        for ballot in voter_preferences:
            for cand in ballot:
                if cand in candidate_votes:
                    candidate_votes[cand] += self.num_candidates - 1 - ballot.index(cand)
                else:
                    candidate_votes[cand] =  self.num_candidates - 1 - ballot.index(cand)
        
        # Find the candidate with the highest total score
        highest_vote = max(candidate_votes.values())
        winning_candidate_list = list(filter(lambda x: candidate_votes[x] == highest_vote, candidate_votes))
        winning_candidate = random.choice(winning_candidate_list)

        return winning_candidate


    def compute_approval_winner(self, voter_ballot_iter):
        approval_dict = {}

        for cand in range(self.num_candidates):
            approval_dict[cand] = 0

        for approval_vector in voter_ballot_iter.values():
            for cand in range(self.num_candidates):
                approval_dict[cand] += approval_vector[cand]

        max_approval = max(approval_dict.values())
        # if more than one ballot have highest rewards then choose one randomly
        top_cand_list = list(filter(lambda x: approval_dict[x] == max_approval, approval_dict))
        winner = random.choice(top_cand_list)

        return winner
    

    def compute_copeland_winner(self, voter_ballot_iter):
        pair_wise_wins = {}
        pair_wise_losses = {}
        copeland_score = {}

        for cand in range(self.num_candidates):
            pair_wise_wins[cand] = 0
            pair_wise_losses[cand] = 0
            copeland_score[cand] = 0

        pair_wise_combinations = combinations(range(self.num_candidates), 2)
        for pair in pair_wise_combinations:
            for ballot in voter_ballot_iter.values():
                if ballot.index(pair[0]) > ballot.index(pair[1]):
                    pair_wise_wins[pair[1]] += 1
                    pair_wise_losses[pair[0]] += 1
                else:
                    pair_wise_wins[pair[0]] += 1
                    pair_wise_losses[pair[1]] += 1

        for cand in range(self.num_candidates):
            copeland_score[cand] = pair_wise_wins[cand] - pair_wise_losses[cand]
        highest_vote = max(copeland_score.values())
        winning_candidate_list = list(filter(lambda x: copeland_score[x] == highest_vote, copeland_score))
        winning_candidate = random.choice(winning_candidate_list)

        return winning_candidate              


    def compute_committee_approval_winner(self, voter_ballot_iter):
        winning_committee = {}
        for committee in self.all_committee_combinations:
            winning_committee[tuple(committee)] = 0
        
        for ballot in voter_ballot_iter.values():
            for committee in self.all_committee_combinations:
                for cand in committee:
                    if ballot[cand] == 1:
                        winning_committee[tuple(committee)] += 1
                        break

        highest_vote = max(winning_committee.values())
        winning_committee_list = list(filter(lambda x: winning_committee[x] == highest_vote, winning_committee))
        num_ties = len(winning_committee_list)

        ''' 
            Tie-breaking - Dictionary order
            Exanple: winning_committee_list = [[1, 2, 3], [1, 2, 4]]
            winning_committee = [1, 2, 3] - last index tie-breaking
        '''
        winning_committee = None

        if self.tie_breaking_rule == 'dict':
            for i in range(len(winning_committee_list[0])):
                curr_max = -1
                for j in range(len((winning_committee_list))):
                    if winning_committee_list[j][i] > curr_max:
                        curr_max = winning_committee_list[j][i]
                        winning_committee = winning_committee_list[j]
                if winning_committee:
                    break
        elif self.tie_breaking_rule == 'rand':
            winning_committee = random.choice(winning_committee_list)
        return winning_committee, num_ties
    

    def compute_k_borda_winner(self, voter_ballot_iter):
        voter_preferences = voter_ballot_iter.values()
        candidate_votes = {}
        winning_committee = []

        for ballot in voter_preferences:
            for cand in ballot:
                if cand in candidate_votes:
                    candidate_votes[cand] += self.num_candidates - 1 - ballot.index(cand)
                else:
                    candidate_votes[cand] =  self.num_candidates - 1 - ballot.index(cand)
        
        # Find the top k candidates with the highest total scores
        winning_committee = sorted(candidate_votes, key = candidate_votes.get, reverse=True)[:self.committee_size]
        return winning_committee
    

    def compute_k_plurality_winner(self, voter_ballot_iter):
        voter_preferences = voter_ballot_iter.values()
        candidate_votes = {}
        winning_committee = []

        for cand in voter_preferences:
            if cand in candidate_votes:
                candidate_votes[cand] += 1
            else:
                candidate_votes[cand] = 1

        winning_committee = sorted(candidate_votes, key = candidate_votes.get, reverse=True)[:self.committee_size]
        return winning_committee
    

    def compute_k_anti_plurality_winner(self, voter_ballot_iter):
        voter_preferences = voter_ballot_iter.values()
        candidate_votes = {}
        winning_committee = []

        for cand in voter_preferences:
            if cand in candidate_votes:
                candidate_votes[cand] += 1
            else:
                candidate_votes[cand] = 1

        winning_committee = sorted(candidate_votes, key = candidate_votes.get, reverse=False)[:self.committee_size]
        return winning_committee
    

    def compute_chamberlin_courant_winner(self, voter_ballot_iter):
        winning_committee = []
        winning_committee = chamberlin_courant_borda_utility(voter_ballot_iter, self.committee_size)
        return winning_committee
    

    def compute_bloc_winner(self, voter_ballot_iter):
        voter_preferences = voter_ballot_iter.values()
        candidate_votes = {}
        winning_committee = []

        for ballot in voter_preferences:
            for cand in ballot[:self.committee_size]:   # assign 1 for each top k candidates in the voter preference 
                if cand in candidate_votes:
                    candidate_votes[cand] += 1
                else:
                    candidate_votes[cand] = 1

        winning_committee = sorted(candidate_votes, key = candidate_votes.get, reverse=True)[:self.committee_size]

        return winning_committee
    

    def compute_monroe_winner(self, voter_ballot_iter):
        winning_committee = []
        winning_committee = monroe_borda_utility(voter_ballot_iter, self.committee_size)
        return winning_committee


    def compute_stv_winner(self, voter_ballot_iter):
        voter_preferences = voter_ballot_iter.values()
        voter_preferences_list = []
        for ballot in voter_preferences:
            voter_preferences_list.append(list(ballot))

        winning_committee = []
        candidates = list(range(self.num_candidates))

        while len(voter_preferences_list[0]) > self.committee_size:
            candidate_votes = {}
            for cand in candidates:
                candidate_votes[cand] = 0

            for ballot in voter_preferences_list:
                # assign 1 for top candidate in the voter preference 
                candidate_votes[ballot[0]] += 1

            losing_cand = min(candidate_votes, key = candidate_votes.get)
            candidates.remove(losing_cand)
            for voter in range(self.num_voters):
                voter_preferences_list[voter].remove(losing_cand)

        winning_committee = candidates
        return winning_committee


    def compute_pav_winner(self, voter_ballot_iter):
        winning_committee = []

        approval_dict = {}
        for v in range(self.num_voters):
            approval_dict[v] = {}
            for c in range(self.num_candidates):
                if c in voter_ballot_iter[v][:self.approval_count]:
                    approval_dict[v][c] = 1
                else:
                    approval_dict[v][c] = 0
        winning_committee = pav_utility(voter_ballot_iter, self.committee_size, approval_dict)
        return winning_committee
    

    def compute_winner(self, voter_ballot_iter):
        winner = []
        if self.voting_setting == 1:    # committee voting setting
            if self.voting_rule == 'approval':
                winner, num_ties = self.compute_committee_approval_winner(voter_ballot_iter)
                return winner, num_ties
            elif self.voting_rule == 'plurality':
                winner = self.compute_k_plurality_winner(voter_ballot_iter)
            elif self.voting_rule == 'anti_plurality':
                winner = self.compute_k_anti_plurality_winner(voter_ballot_iter)
            elif self.voting_rule == 'borda' or self.voting_rule == 'borda_top_cand':
                winner = self.compute_k_borda_winner(voter_ballot_iter)
            elif self.voting_rule == 'chamberlin_courant':
                winner = self.compute_chamberlin_courant_winner(voter_ballot_iter)
            elif self.voting_rule == 'bloc':
                winner = self.compute_bloc_winner(voter_ballot_iter)
            elif self.voting_rule == 'monroe':
                winner = self.compute_monroe_winner(voter_ballot_iter)
            elif self.voting_rule == 'stv':
                winner = self.compute_stv_winner(voter_ballot_iter)
            elif self.voting_rule == 'pav':
                winner = self.compute_pav_winner(voter_ballot_iter)

        else:                           # single winner setting
            if self.voting_rule == 'plurality':
                winner = self.compute_plurality_winner(voter_ballot_iter)
            elif self.voting_rule == 'borda':
                winner = self.compute_borda_winner(voter_ballot_iter)
            elif self.voting_rule == 'approval':
                winner = self.compute_approval_winner(voter_ballot_iter)
            elif self.voting_rule == 'copeland':
                winner = self.compute_copeland_winner(voter_ballot_iter)
        return winner


    def compute_welfare(self, voter_ballot_iter, winner):

        welfare_dict = {x:0 for x in range(self.num_voters)}
        
        # single winner setting, comupte welfare using borda utility
        if self.voting_setting == 0:
            for voter in range(self.num_voters):
                actual_voter_ballot = self.full_voting_profile[voter]
                welfare_dict[voter] = self.welfare_scoring_vector[actual_voter_ballot.index(winner)] # compute the score for the winner using scoring vector and voter i's preferences
        
        # committee voting setting
        else:
            for voter in range(self.num_voters):
                actual_voter_ballot = self.full_voting_profile[voter]
                for cand in winner:
                    welfare_dict[voter] += self.welfare_scoring_vector[actual_voter_ballot.index(cand)]
        
            # # welfare rule for k-approval - if the chosen candidate is approved by the voter, 
            # # the welfare of the voter will be cummulative borda scores
            # if self.voting_rule == 'approval':
            #     for voter in range(self.num_voters):
            #         actual_voter_ballot = self.full_voting_profile[voter][:self.approval_count]
            #         for cand in actual_voter_ballot:
            #             if cand in winner:
            #                 welfare_dict[voter] += self.welfare_scoring_vector[actual_voter_ballot.index(cand)]
            
            # # welfare rule for k-borda - compute the cummulative borda scores for the chosen candidate in the ranked preference
            # elif self.voting_rule == 'borda':
            #     for voter in range(self.num_voters):
            #         actual_voter_ballot = self.full_voting_profile[voter]
            #         for cand in winner:
            #             welfare_dict[voter] += self.welfare_scoring_vector[actual_voter_ballot.index(cand)]
            
            # welfare rule for borda top cand - compute the borda score for the highest ranked 
            # (highest borda score) candidate in the committee
            # elif self.voting_rule == 'borda_top_cand':
            #     for voter in range(self.num_voters):
            #         actual_voter_ballot = self.full_voting_profile[voter]
            #         welfare_dict[voter] = self.welfare_scoring_vector[actual_voter_ballot.index(winner[0])]
            
            # # welfare rule for k-plurality - if top cand in actual voter ballot is part of winning committiee
            # #  voter will get score 1 else 0
            # elif self.voting_rule == 'plurality' or self.voting_rule == 'anti_plurality':
            #     for voter in range(self.num_voters):
            #         # actual_top_cand = self.full_voting_profile[voter][0]
            #         # if actual_top_cand in winner:
            #         #     welfare_dict[voter] = 1

            #         # with borda utility score
            #         actual_voter_ballot = self.full_voting_profile[voter]
            #         actual_top_cand = actual_voter_ballot[0]
            #         if actual_top_cand in winner:
            #             welfare_dict[voter] = self.welfare_scoring_vector[actual_voter_ballot.index(actual_top_cand)]
            
            # # welfare rule for chamberlin_courant: compute the cummulative borda scores 
            # # for the chosen candidate in the ranked preference
            # elif self.voting_rule == 'chamberlin_courant':
            #     for voter in range(self.num_voters):
            #         actual_voter_ballot = self.full_voting_profile[voter]
            #         for cand in winner:
            #             welfare_dict[voter] += self.welfare_scoring_vector[actual_voter_ballot.index(cand)]
        
            # # welfare rule for bloc: compute the total plurality score for all the candidates in the committee
            # elif self.voting_rule == 'bloc':
            #     for voter in range(self.num_voters):
            #         actual_voter_ballot = self.full_voting_profile[voter]
            #         for cand in winner:
            #             if cand in actual_voter_ballot[:self.committee_size]:
            #                 welfare_dict[voter] += self.welfare_scoring_vector[actual_voter_ballot.index(cand)]
            
            # # welfare rule for monroe: compute the cummulative borda scores for the chosen candidate in the ranked preference
            # elif self.voting_rule == 'monroe':
            #     for voter in range(self.num_voters):
            #         actual_voter_ballot = self.full_voting_profile[voter]
            #         for cand in winner:
            #             welfare_dict[voter] += self.welfare_scoring_vector[actual_voter_ballot.index(cand)]
        
            # elif self.voting_rule == 'stv':
            #     for voter in range(self.num_voters):
            #         actual_top_cand = self.full_voting_profile[voter][0]
            #         if actual_top_cand in winner:
            #             welfare_dict[voter] = self.welfare_scoring_vector[actual_voter_ballot.index(actual_top_cand)]

            # elif self.voting_rule == 'pav':
            #     for voter in range(self.num_voters):
            #         actual_voter_ballot = self.full_voting_profile[voter][:self.approval_count]
            #         for cand in actual_voter_ballot:
            #             if cand in winner:
            #                 welfare_dict[voter] += self.welfare_scoring_vector[actual_voter_ballot.index(cand)]
            
            # else:
            #     for voter in range(self.num_voters):
            #         actual_voter_ballot = self.full_voting_profile[voter]
            #         for cand in winner:
            #             welfare_dict[voter] += self.welfare_scoring_vector[actual_voter_ballot.index(cand)]
        
        return welfare_dict


    def update_rewards(self, voter_ballot_dict, voter_ballot_iter, welfare_dict):
        # update reward and count in voter_ballot_dict for every voter and the 'arm' they pick with the welfare, i.e., reward in the current iteration
        for voter in voter_ballot_iter.keys():
            if self.voting_rule == 'plurality' or self.voting_rule == 'anti_plurality':
                ballot = voter_ballot_iter[voter]
            elif self.voting_rule == 'approval':
                ballot = tuple(voter_ballot_iter[voter])
            else: # self.voting_rule == 'borda' or 'borda_top_cand' or 'copeland' or 'chamberlin_courant' or 'bloc':
                ballot = tuple(voter_ballot_iter[voter])

            voter_ballot_dict[voter]['count'][ballot] += 1
            voter_ballot_dict[voter]['reward'][ballot] = (voter_ballot_dict[voter]['reward'][ballot] + welfare_dict[voter])
            # print('came ', voter_ballot_dict[voter]['reward'][ballot])
        return voter_ballot_dict


    # train the model for given number of iterations and given training explore_criteria
    def train(self, explore_criteria, single_iterative_voting):

        borda_scores_arr = []   # list of borda scores at given n iterations
        welfare_dict_list = []  # list of welfare dictionaries of voters
        iter_winners_list = [] # list of winners for n iterations
        voter_ballot_iter_list = [] # list of votter ballots for n iterations
        num_ties_list = []

        voter_top_candidates = []
        for voter in range(self.num_voters):
            voter_top_candidates.append(self.full_voting_profile[voter][0])

        # append all votes to list, create a list of dictionary for all voters having votes of each voting round
        voter_ballot_dict = {}      # dictionary containing each voter ballot list throughout the iterations

        for voter in range(self.num_voters):
            voter_ballot_dict[voter] = {}
            voter_ballot_dict[voter]['reward'] = {}
            voter_ballot_dict[voter]['count'] = {}

            if self.voting_rule == 'plurality' or self.voting_rule == 'anti_plurality':
                for cand in range(self.num_candidates):
                    voter_ballot_dict[voter]['reward'][cand] = 0
                    voter_ballot_dict[voter]['count'][cand] = 0

            elif self.voting_rule == 'approval':
                for comb in self.all_approval_combinations:
                    voter_ballot_dict[voter]['reward'][tuple(comb)] = 0
                    voter_ballot_dict[voter]['count'][tuple(comb)] = 0

            # self.voting_rule == 'borda' or 'borda_top_cand' or 'copeland' or 'chamberlin_courant' or 'bloc' or 'pav' or 'stv':
            else:
                for cand in list(permutations(range(self.num_candidates))):
                    voter_ballot_dict[voter]['reward'][tuple(cand)] = 0
                    voter_ballot_dict[voter]['count'][tuple(cand)] = 0

        # for 0th iteration
        voter_ballot_iter = {}        # dictionary containing voter and their corresponding ballot

        for voter_i in range(self.num_voters):
            if self.voting_rule == 'plurality' or self.voting_rule == 'anti_plurality':
                cand = self.full_voting_profile[voter_i][0]  # initialy asign this dictionary with true top candidate of voters
            elif self.voting_rule == 'approval':
                cand = [0]*self.num_candidates
                for c in self.full_voting_profile[voter_i][ : self.approval_count]:  # true approval vector of voters, first k cands get 1 and rest 0
                    cand[c] = 1
                cand = tuple(cand)
            else: # self.voting_rule == 'borda' or 'borda_top_cand' or 'copeland' or 'chamberlin_courant' or 'bloc':
                cand = tuple(self.full_voting_profile[voter_i])  # initialy asign this dictionary with true preferences of voters
            
            voter_ballot_iter[voter_i] = cand
        
        voter_ballot_iter_dict = copy.deepcopy(voter_ballot_iter)
        voter_ballot_iter_list.append(voter_ballot_iter_dict)

        if self.voting_setting == 1 and self.voting_rule == 'approval':
            winning_candidate, num_ties = self.compute_winner(voter_ballot_iter)
            num_ties_list.append(num_ties)
        else:
            winning_candidate = self.compute_winner(voter_ballot_iter)
        
        iter_winners_list.append(winning_candidate)

        welfare_dict = self.compute_welfare(voter_ballot_iter, winning_candidate)
        self.update_rewards(voter_ballot_dict, voter_ballot_iter, welfare_dict)

        winning_borda_score = 0
        for voter in range(self.num_voters):
            if self.voting_rule == 'plurality' or self.voting_rule == 'anti_plurality':
                ballot_iter = voter_ballot_iter[voter]
            elif self.voting_rule == 'approval':
                ballot_iter = tuple(voter_ballot_iter[voter])
            else: # self.voting_rule == 'borda' or 'borda_top_cand' or 'copeland' or 'chamberlin_courant' or 'bloc':
                ballot_iter = tuple(voter_ballot_iter[voter])

            if (voter_ballot_dict[voter]['count'][ballot_iter]):
                winning_borda_score += voter_ballot_dict[voter]['reward'][ballot_iter] / voter_ballot_dict[voter]['count'][ballot_iter]
        print(winning_borda_score, winning_candidate)
        borda_scores_arr.append(winning_borda_score)
        welfare_dict_list.append(welfare_dict)

        # Run the training loop for iterations - 1 times with pick arm method exploration and exploitation method
        for iter in range(1, self.iterations - 1):
            if not single_iterative_voting:
                # run one voting cycle where all the voters cast their vote using pick_arm method and give the their top candidate
                self.get_vote(explore_criteria, voter_ballot_dict, voter_ballot_iter, voter=None)

            else:
                # In each iteration pick one voter randomly and they will use the pick arm method to vote
                manupilating_voter = random.choice([i for i in range(self.num_voters)])
                self.get_vote(explore_criteria, voter_ballot_dict, voter_ballot_iter, voter=manupilating_voter)

            voter_ballot_iter_dict = copy.deepcopy(voter_ballot_iter)
            voter_ballot_iter_list.append(voter_ballot_iter_dict)

            if self.voting_setting == 1 and self.voting_rule == 'approval':
                winning_candidate, num_ties = self.compute_winner(voter_ballot_iter)
                num_ties_list.append(num_ties)
            else:
                winning_candidate = self.compute_winner(voter_ballot_iter)
            iter_winners_list.append(winning_candidate)
            
            welfare_dict = self.compute_welfare(voter_ballot_iter, winning_candidate)
            self.update_rewards(voter_ballot_dict, voter_ballot_iter, welfare_dict)

            # if iter % self.batch == 0:
            winning_borda_score = 0
            # compute the sum of rewards experienced by the voters fot this winning candidate
            for voter in range(self.num_voters):
                if self.voting_rule == 'plurality' or self.voting_rule == 'anti_plurality':
                    ballot_iter = voter_ballot_iter[voter]
                elif self.voting_rule == 'approval':
                    ballot_iter = tuple(voter_ballot_iter[voter])
                else: # self.voting_rule == 'borda' or 'borda_top_cand' or 'copeland' or 'chamberlin_courant' or 'bloc':
                    ballot_iter = tuple(voter_ballot_iter[voter])

                # highest_reward_cand = max(voter_ballot_dict[voter]['reward'], key=voter_ballot_dict[voter]['reward'].get)
                if (voter_ballot_dict[voter]['count'][ballot_iter]):
                    winning_borda_score += voter_ballot_dict[voter]['reward'][ballot_iter] / voter_ballot_dict[voter]['count'][ballot_iter]
            
            borda_scores_arr.append(winning_borda_score)
            welfare_dict_list.append(welfare_dict)
        
        # Get the results for explotation in last iteration
        for voter in range(self.num_voters):
            curr_reward = {}
            for cand in voter_ballot_dict[voter]['reward'].keys():
                if voter_ballot_dict[voter]['count'][cand] > 0:
                    curr_reward[cand] = voter_ballot_dict[voter]['reward'][cand] / voter_ballot_dict[voter]['count'][cand]
                else:
                    curr_reward[cand] = voter_ballot_dict[voter]['reward'][cand]
            max_reward = max(curr_reward.values())
            top_ballot_list = list(filter(lambda x: curr_reward[x] == max_reward, curr_reward))
            top_ballot = random.choice(top_ballot_list)

            voter_ballot_iter[voter_i] = top_ballot
        
        voter_ballot_iter_dict = copy.deepcopy(voter_ballot_iter)
        voter_ballot_iter_list.append(voter_ballot_iter)

        if self.voting_setting == 1 and self.voting_rule == 'approval':
            winning_candidate, num_ties = self.compute_winner(voter_ballot_iter)
            num_ties_list.append(num_ties)
        else:
            winning_candidate = self.compute_winner(voter_ballot_iter)

        iter_winners_list.append(winning_candidate)

        welfare_dict = self.compute_welfare(voter_ballot_iter, winning_candidate)
        self.update_rewards(voter_ballot_dict, voter_ballot_iter, welfare_dict)

        winning_borda_score = 0
        # compute the sum of rewards experienced by the voters fot this winning candidate
        for voter in range(self.num_voters):
            if self.voting_rule == 'plurality' or self.voting_rule == 'anti_plurality':
                ballot_iter = voter_ballot_iter[voter]
            elif self.voting_rule == 'approval':
                ballot_iter = tuple(voter_ballot_iter[voter])
            else: # self.voting_rule == 'borda' or 'borda_top_cand' or 'copeland' or 'chamberlin_courant' or 'bloc':
                ballot_iter = tuple(voter_ballot_iter[voter])

            # highest_reward_cand = max(voter_ballot_dict[voter]['reward'], key=voter_ballot_dict[voter]['reward'].get)
            if (voter_ballot_dict[voter]['count'][ballot_iter]):
                winning_borda_score += voter_ballot_dict[voter]['reward'][ballot_iter] / voter_ballot_dict[voter]['count'][ballot_iter]
        
        borda_scores_arr.append(winning_borda_score)
        welfare_dict_list.append(welfare_dict)

        return borda_scores_arr, welfare_dict_list, iter_winners_list, voter_ballot_iter_list, num_ties_list

