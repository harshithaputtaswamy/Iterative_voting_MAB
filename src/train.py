import json
import random
import numpy as np
from model import model


# train the model for given number of iterations and given training algo
def train(iterations, batch, algo, epsilon, parsed_soc_data, single_iterative_voting, grad_epsilon, epsilon_final, epsilon_decay):
    actual_mean = parsed_soc_data["borda_scores"]   #the actual borda score of each candidate
    num_candidates = parsed_soc_data["num_candidates"]
    num_voters = parsed_soc_data["num_voters"]
    full_voting_profile = parsed_soc_data["full_voting_profile"]
    flattened_voting_profile = parsed_soc_data["flattened_voting_profile"]

    borda_scores_arr = []   # list of borda scores at given n iterations
    # epsilon_decay = np.power(epsilon_final / epsilon, 1.0 / iterations)


    full_voting_profile = []
    for ballet, num in flattened_voting_profile.items():
        for i in range(num):
            full_voting_profile.append(json.loads(ballet))
    
    # print(full_voting_profile)
    voter_top_candidates = []
    for voter in range(num_voters):
        voter_top_candidates.append(full_voting_profile[voter][0])

    # get the winning ballet from the borda scores of candidates
    winning_ballot = [int(i) for i in list(dict(sorted(actual_mean.items(), key=lambda item: item[1], reverse=True)).keys())]
    # print("winning_ballot ", winning_ballot)


    # generate borda score dictionary for candidates and initial with 0
    reward = {}
    for cand in range(num_candidates):
        reward[cand] = 0

    agent = model(epsilon, num_candidates, num_voters)

    # append all votes to list, create a list of dictionary for all voters having votes of each voting round
    voter_ballet_dict = {}      # dictionary containing each voter ballet list throughout the iterations

    for voter in range(num_voters):
        voter_ballet_dict[voter] = {}
        voter_ballet_dict[voter]["reward"] = {}
        voter_ballet_dict[voter]["count"] = {}
        for cand in range(num_candidates):
            voter_ballet_dict[voter]["reward"][cand] = 0
            voter_ballet_dict[voter]["count"][cand] = 0

    voter_ballet_iter = {}        # dictionary containing voter and their corresponding top candidate
    
    n_iter_reward = {}
    for voter in range(num_voters):
        n_iter_reward[voter] = 0

    for iter in range(iterations):
        candidate_votes = {}        # dictionary containing number votes per candidate

        if not single_iterative_voting:
            # run one voting cycle where all the voters cast their vote using pick_arm method and give the their top candidate
            for voter in range(num_voters):
                top_candidate = agent.pick_arm(algo, reward, voter, voter_ballet_dict[voter], grad_epsilon, epsilon_final, epsilon_decay)
                voter_ballet_iter[voter] = top_candidate
                voter_ballet_dict[voter]["count"][top_candidate] += 1
        else:
            # In each iteration pick one voter randomly and they will use the pick arm method to vote
            manupilating_voter = random.choice([i for i in range(num_voters)])
            top_candidate = agent.pick_arm(algo, reward, manupilating_voter, voter_ballet_dict[manupilating_voter], grad_epsilon, epsilon_final, epsilon_decay)
            voter_ballet_iter[manupilating_voter] = top_candidate
            voter_ballet_dict[manupilating_voter]["count"][top_candidate] += 1

            voter_list = [i for i in range(num_voters)]
            voter_list.remove(manupilating_voter)
            for voter in voter_list:
                cand = full_voting_profile[voter][0]  # initialy asign this dictionary with true preferences of voters
                voter_ballet_iter[voter] = cand
                voter_ballet_dict[voter]["count"][cand] += 1

# TODO: compute_winner(), compute_wellfare(), update_reward()

        # update the candidate votes dictionary
        for top_candidate in voter_ballet_iter.values():
            if top_candidate in candidate_votes:
                candidate_votes[top_candidate] += 1
            else:
                candidate_votes[top_candidate] = 1

        # get the borda score by comapring the winning candidate and the actual voter preference, update the reward for each voter for their voted candidate
        for voter in range(num_voters):
            actual_voter_ballet = full_voting_profile[voter]
            top_cand = voter_ballet_iter[voter]
            borda_score = num_candidates - 1 - actual_voter_ballet.index(top_cand)
            voter_ballet_dict[voter]["reward"][top_cand] = (voter_ballet_dict[voter]["reward"][top_cand] + borda_score)

        if iter % batch == 0:
            # print(agent.epsilon)
            # get the list of winning candidates, if there are more than two winning candidates chooce one randomly (tie breaking)
            highest_vote = max(candidate_votes.values())
            winning_candidate_list = list(filter(lambda x: candidate_votes[x] == highest_vote, candidate_votes))
            winning_candidate = random.choice(winning_candidate_list)

            winning_borda_score = 0
            # compute the sum of rewards experienced by the voters fot this winning candidate
            for voter in range(num_voters):
                if (voter_ballet_dict[voter]["count"][winning_candidate]):
                    winning_borda_score += voter_ballet_dict[voter]["reward"][winning_candidate] / voter_ballet_dict[voter]["count"][winning_candidate]

            borda_scores_arr.append(winning_borda_score)
            # print("winning_cansdidate ", winning_candidate)
            # print("final_round_reward_cand ", borda_scores_arr)

        # print("winner profile ", candidate_votes)

    return borda_scores_arr


# train(10000, 1, 0.5, parsed_soc_data)