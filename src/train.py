import json
import random
import numpy as np
from itertools import permutations, combinations


class train_model():
    def __init__(self, agent, iterations, batch, committee_size, voting_rule, voting_setting, parsed_soc_data, epsilon, grad_epsilon, epsilon_final, epsilon_decay, approval_count=0):
        self.agent = agent
        self.parsed_soc_data = parsed_soc_data
        self.num_candidates = parsed_soc_data["num_candidates"]
        self.num_voters = parsed_soc_data["num_voters"]
        self.flattened_voting_profile = parsed_soc_data["flattened_voting_profile"]
        self.full_voting_profile = parsed_soc_data["full_voting_profile"]

        self.iterations = iterations
        self.batch = batch
        self.grad_epsilon = grad_epsilon
        self.epsilon = epsilon
        self.epsilon_final = epsilon_final
        self.epsilon_decay = epsilon_decay

        self.committee_size = committee_size
        self.voting_rule = voting_rule
        self.voting_setting = voting_setting

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
        winning_committee = random.choice(winning_committee_list)

        return winning_committee
    

    def compute_k_borda_winner(self, voter_ballot_iter):
        voter_preferences = voter_ballot_iter.values()
        candidate_votes = {}

        for ballot in voter_preferences:
            for cand in ballot:
                if cand in candidate_votes:
                    candidate_votes[cand] += self.num_candidates - 1 - ballot.index(cand)
                else:
                    candidate_votes[cand] =  self.num_candidates - 1 - ballot.index(cand)
        
        # Find the top k candidates with the highest total scores
        winning_committee = sorted(candidate_votes, key = candidate_votes.get, reverse=True)[:self.committee_size]
        print(winning_committee)

        return winning_committee


    def compute_winner(self, voter_ballot_iter):
        winner = []
        if self.voting_setting == 1:    # committee voting setting
            if self.voting_rule == 'approval':
                winner = self.compute_committee_approval_winner(voter_ballot_iter)
            elif self.voting_rule == 'borda' or self.voting_rule == 'borda_top_cand':
                winner = self.compute_k_borda_winner(voter_ballot_iter)

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

        if self.voting_setting == 0:    # single winner setting, comupte welfare using borda utility
            for voter in range(self.num_voters):
                actual_voter_ballot = self.full_voting_profile[voter]
                welfare_dict[voter] = self.welfare_scoring_vector[actual_voter_ballot.index(winner)] # compute the score for the winner using scoring vector and voter i's preferences

        else:                           # committee voting setting
            if self.voting_rule == 'approval':      # welfare rule for k-approval - if the chosen candidate is approved by the voter, the welfare of the voter will be 1 else 0
                for voter in range(self.num_voters):
                    actual_voter_ballot = self.full_voting_profile[voter][:self.approval_count]
                    for cand in actual_voter_ballot:
                        if cand in winner:
                            welfare_dict[voter] = 1
                            break
            elif self.voting_rule == 'borda':     # welfare rule for k-borda - compute the cummulative borda scores for the chosen candidate in the ranked preference
                for voter in range(self.num_voters):
                    actual_voter_ballot = self.full_voting_profile[voter]
                    for cand in winner:
                        welfare_dict[voter] += self.welfare_scoring_vector[actual_voter_ballot.index(cand)]
            elif self.voting_rule == 'borda_top_cand':     # welfare rule - compute the borda score for the highest ranked (highest borda score) candidate in the committee
                for voter in range(self.num_voters):
                    actual_voter_ballot = self.full_voting_profile[voter]
                    welfare_dict[voter] = self.welfare_scoring_vector[actual_voter_ballot.index(winner[0])]
        return welfare_dict


    def update_rewards(self, voter_ballot_dict, voter_ballot_iter, welfare_dict):
        # update reward and count in voter_ballot_dict for every voter and the "arm" they pick with the welfare, i.e., reward in the current iteration
        for voter in voter_ballot_iter.keys():
            if self.voting_rule == "plurality":
                ballot = voter_ballot_iter[voter]
            elif self.voting_rule == "approval":
                ballot = tuple(voter_ballot_iter[voter])
            elif self.voting_rule == "borda" or self.voting_rule == 'borda_top_cand':
                ballot = tuple(voter_ballot_iter[voter])
            elif self.voting_rule == 'copeland':
                ballot = tuple(voter_ballot_iter[voter])

            voter_ballot_dict[voter]["count"][ballot] += 1
            voter_ballot_dict[voter]["reward"][ballot] = (voter_ballot_dict[voter]["reward"][ballot] + welfare_dict[voter])
            # print("came ", voter_ballot_dict[voter]["reward"][ballot])
        return voter_ballot_dict


    # train the model for given number of iterations and given training explore_criteria
    def train(self, explore_criteria, single_iterative_voting):

        borda_scores_arr = []   # list of borda scores at given n iterations
        welfare_dict_list = []  # list of welfare dictionaries of voters
        iter_winners_list = [] # list of winners for n iterations
        voter_ballot_iter_list = [] # list of votter ballots for n iterations

        voter_top_candidates = []
        for voter in range(self.num_voters):
            voter_top_candidates.append(self.full_voting_profile[voter][0])

        # append all votes to list, create a list of dictionary for all voters having votes of each voting round
        voter_ballot_dict = {}      # dictionary containing each voter ballot list throughout the iterations

        for voter in range(self.num_voters):
            voter_ballot_dict[voter] = {}
            voter_ballot_dict[voter]["reward"] = {}
            voter_ballot_dict[voter]["count"] = {}

            if self.voting_rule == 'plurality':
                for cand in range(self.num_candidates):
                    voter_ballot_dict[voter]["reward"][cand] = 0
                    voter_ballot_dict[voter]["count"][cand] = 0

            elif self.voting_rule == 'approval':
                for comb in self.all_approval_combinations:
                    voter_ballot_dict[voter]["reward"][tuple(comb)] = 0
                    voter_ballot_dict[voter]["count"][tuple(comb)] = 0

            elif self.voting_rule == 'borda' or self.voting_rule == 'borda_top_cand':
                for cand in list(permutations(range(self.num_candidates))):
                    voter_ballot_dict[voter]["reward"][tuple(cand)] = 0
                    voter_ballot_dict[voter]["count"][tuple(cand)] = 0

            elif self.voting_rule == 'copeland':
                for cand in list(permutations(range(self.num_candidates))):
                    voter_ballot_dict[voter]["reward"][tuple(cand)] = 0
                    voter_ballot_dict[voter]["count"][tuple(cand)] = 0

        # for 0th iteration
        voter_ballot_iter = {}        # dictionary containing voter and their corresponding ballot

        for voter_i in range(self.num_voters):
            if self.voting_rule == 'plurality':
                cand = self.full_voting_profile[voter_i][0]  # initialy asign this dictionary with true top candidate of voters
            elif self.voting_rule == 'borda' or self.voting_rule == 'borda_top_cand':
                cand = tuple(self.full_voting_profile[voter_i])  # initialy asign this dictionary with true preferences of voters
            elif self.voting_rule == 'copeland':
                cand = tuple(self.full_voting_profile[voter_i])
            elif self.voting_rule == 'approval':
                cand = [0]*self.num_candidates
                for c in self.full_voting_profile[voter_i][ : self.approval_count]:  # true approval vector of voters, first k cands get 1 and rest 0
                    cand[c] = 1
                cand = tuple(cand)
            
            voter_ballot_iter[voter_i] = cand
        
        voter_ballot_iter_list.append(voter_ballot_iter)

        winning_candidate = self.compute_winner(voter_ballot_iter)
        iter_winners_list.append(winning_candidate)

        welfare_dict = self.compute_welfare(voter_ballot_iter, winning_candidate)
        self.update_rewards(voter_ballot_dict, voter_ballot_iter, welfare_dict)

        winning_borda_score = 0
        for voter in range(self.num_voters):
            if self.voting_rule == "plurality":
                ballot_iter = voter_ballot_iter[voter]
            elif self.voting_rule == "approval":
                ballot_iter = tuple(voter_ballot_iter[voter])
            elif self.voting_rule == "borda" or self.voting_rule == 'borda_top_cand':
                ballot_iter = tuple(voter_ballot_iter[voter])
            elif self.voting_rule == "copeland":
                ballot_iter = tuple(voter_ballot_iter[voter])

            if (voter_ballot_dict[voter]["count"][ballot_iter]):
                winning_borda_score += voter_ballot_dict[voter]["reward"][ballot_iter] / voter_ballot_dict[voter]["count"][ballot_iter]

        borda_scores_arr.append(winning_borda_score)
        welfare_dict_list.append(welfare_dict)

        for iter in range(1, self.iterations):
            if not single_iterative_voting:
                # run one voting cycle where all the voters cast their vote using pick_arm method and give the their top candidate
                self.get_vote(explore_criteria, voter_ballot_dict, voter_ballot_iter, voter=None)

            else:
                # In each iteration pick one voter randomly and they will use the pick arm method to vote
                manupilating_voter = random.choice([i for i in range(self.num_voters)])
                self.get_vote(explore_criteria, voter_ballot_dict, voter_ballot_iter, voter=manupilating_voter)

            voter_ballot_iter_list.append(voter_ballot_iter)

            winning_candidate = self.compute_winner(voter_ballot_iter)
            iter_winners_list.append(winning_candidate)

            welfare_dict = self.compute_welfare(voter_ballot_iter, winning_candidate)
            self.update_rewards(voter_ballot_dict, voter_ballot_iter, welfare_dict)

            # if iter % self.batch == 0:
            winning_borda_score = 0
            # compute the sum of rewards experienced by the voters fot this winning candidate
            for voter in range(self.num_voters):
                if self.voting_rule == "plurality":
                    ballot_iter = voter_ballot_iter[voter]
                elif self.voting_rule == "approval":
                    ballot_iter = tuple(voter_ballot_iter[voter])
                elif self.voting_rule == "borda" or self.voting_rule == 'borda_top_cand':
                    ballot_iter = tuple(voter_ballot_iter[voter])
                elif self.voting_rule == "copeland":
                    ballot_iter = tuple(voter_ballot_iter[voter])
                # highest_reward_cand = max(voter_ballot_dict[voter]["reward"], key=voter_ballot_dict[voter]["reward"].get)
                if (voter_ballot_dict[voter]["count"][ballot_iter]):
                    winning_borda_score += voter_ballot_dict[voter]["reward"][ballot_iter] / voter_ballot_dict[voter]["count"][ballot_iter]
            borda_scores_arr.append(winning_borda_score)
            welfare_dict_list.append(welfare_dict)

        return borda_scores_arr, welfare_dict_list, iter_winners_list, voter_ballot_iter_list

