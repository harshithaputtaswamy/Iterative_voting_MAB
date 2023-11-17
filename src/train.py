import json
import random
import numpy as np
from model import model
from itertools import permutations


class train_model():
    def __init__(self, iterations, batch, parsed_soc_data, epsilon, grad_epsilon, epsilon_final, epsilon_decay):
        self.iterations = iterations
        self.batch = batch
        self.grad_epsilon = grad_epsilon
        self.epsilon = epsilon
        self.epsilon_final = epsilon_final
        self.epsilon_decay = epsilon_decay
        self.voting_rule = 'plurality'

        self.parsed_soc_data = parsed_soc_data
        self.actual_mean = parsed_soc_data["borda_scores"]   #the actual borda score of each candidate
        self.num_candidates = parsed_soc_data["num_candidates"]
        self.num_voters = parsed_soc_data["num_voters"]
        self.flattened_voting_profile = parsed_soc_data["flattened_voting_profile"]
        self.full_voting_profile = []
        for ballot, num in self.flattened_voting_profile.items():
            for _ in range(num):
                self.full_voting_profile.append(json.loads(ballot))
    

    def get_vote(self, agent, explore_criteria, voter_ballot_dict, voter_ballot_iter, voter = None):
        if voter is None:
            for voter_i in range(self.num_voters):
                chosen_ballot = agent.pick_arm(explore_criteria, voter_ballot_dict[voter_i], self.voting_rule, self.grad_epsilon, self.epsilon_final, self.epsilon_decay)
                voter_ballot_iter[voter_i] = chosen_ballot
        else:
            for voter_i in range(self.num_voters):
                if self.voting_rule == 'plurality':
                    cand = self.full_voting_profile[voter_i][0]  # initialy asign this dictionary with true preferences of voters
                elif self.voting_rule == 'borda':
                    cand = self.full_voting_profile[voter_i]  # initialy asign this dictionary with true preferences of voters

                voter_ballot_iter[voter_i] = cand
                # print("cand", cand)

            chosen_ballot = agent.pick_arm(explore_criteria, voter_ballot_dict[voter], self.voting_rule, self.grad_epsilon, self.epsilon_final, self.epsilon_decay)
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

        # print("winning_cansdidate ", winning_candidate)
        # print("final_round_reward_cand ", borda_scores_arr)

        return winning_candidate


    def compute_borda_winner(self, voter_ballot_iter):
        # print(voter_ballot_iter)
        voter_preferences = voter_ballot_iter.values()
        scoring_vector = list(range(self.num_candidates - 1, -1, -1))

        scores = [0] * self.num_candidates
        # print("voter_preferences", voter_preferences)
        for prefs in voter_preferences:
            for i, candidate in enumerate(prefs):
                scores[candidate] += scoring_vector[i]

        # Find the candidate with the highest total score
        winner = max(range(self.num_candidates), key=lambda candidate: scores[candidate])

        return winner


    def compute_winner(self, voter_ballot_iter):
        if self.voting_rule == 'plurality':
            winner = self.compute_plurality_winner(voter_ballot_iter)
        elif self.voting_rule == 'borda':
            winner = self.compute_borda_winner(voter_ballot_iter)
        return winner


    def compute_welfare(self, voter_ballot_iter, winner):
        welfare_dict = {x:0 for x in range(self.num_voters)}

        if self.voting_rule == 'plurality':
            for voter in range(self.num_voters):
                actual_voter_ballot = self.full_voting_profile[voter]
                # top_cand = voter_ballot_iter[voter]
                welfare_dict[voter] = self.num_candidates - 1 - actual_voter_ballot.index(winner) # compute the score for the winner using scoring vector and voter i's preferences

        elif self.voting_rule == 'borda':
            scoring_vector = list(range(self.num_candidates - 1, -1, -1))
            for voter, voter_preferences in voter_ballot_iter.items():
                welfare_dict[voter] = sum((scoring_vector[j] * (winner == candidate)) for j, candidate in enumerate(voter_preferences))
                print(welfare_dict[voter])
        return welfare_dict


    def update_rewards(self, voter_ballot_dict, voter_ballot_iter, welfare_dict):
        # update reward and count in voter_ballot_dict for every voter and the "arm" they pick with the welfare, i.e., reward in the current iteration
        for voter in voter_ballot_iter.keys():
            ballot = voter_ballot_iter[voter]
            voter_ballot_dict[voter]["count"][ballot] += 1
            voter_ballot_dict[voter]["reward"][ballot] = (voter_ballot_dict[voter]["reward"][ballot] + welfare_dict[voter])
        return voter_ballot_dict


    # train the model for given number of iterations and given training explore_criteria
    def train(self, explore_criteria, single_iterative_voting):

        borda_scores_arr = []   # list of borda scores at given n iterations

        voter_top_candidates = []
        for voter in range(self.num_voters):
            voter_top_candidates.append(self.full_voting_profile[voter][0])

        agent = model(self.epsilon, self.num_candidates, self.num_voters)

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
            elif self.voting_rule == 'borda':
                for cand in list(permutations(list(range(self.num_candidates)))):
                    voter_ballot_dict[voter]["reward"][cand] = 0
                    voter_ballot_dict[voter]["count"][cand] = 0
        
        # print("init voter_ballot_dict", voter_ballot_dict[0]["reward"])

        voter_ballot_iter = {}        # dictionary containing voter and their corresponding ballot

        for iter in range(self.iterations):
            if not single_iterative_voting:
                # run one voting cycle where all the voters cast their vote using pick_arm method and give the their top candidate
                self.get_vote(agent, explore_criteria, voter_ballot_dict, voter_ballot_iter, voter=None)

            else:
                # In each iteration pick one voter randomly and they will use the pick arm method to vote
                manupilating_voter = random.choice([i for i in range(self.num_voters)])
                self.get_vote(agent, explore_criteria, voter_ballot_dict, voter_ballot_iter, voter=manupilating_voter)

            winning_candidate = self.compute_winner(voter_ballot_iter)

            welfare_dict = self.compute_welfare(voter_ballot_iter, winning_candidate)
            self.update_rewards(voter_ballot_dict, voter_ballot_iter, welfare_dict)

            if iter % self.batch == 0:

                winning_borda_score = 0
                # compute the sum of rewards experienced by the voters fot this winning candidate
                for voter in range(self.num_voters):
                    if (voter_ballot_dict[voter]["count"][winning_candidate]):
                        winning_borda_score += voter_ballot_dict[voter]["reward"][winning_candidate] / voter_ballot_dict[voter]["count"][winning_candidate]

                borda_scores_arr.append(winning_borda_score)
                # print("winning_cansdidate ", winning_candidate)
                # print("final_round_reward_cand ", borda_scores_arr)

            # print("winner profile ", candidate_votes)

        return borda_scores_arr


# train(10000, 1, 0.5, parsed_soc_data)