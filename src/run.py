import json
import random
import itertools
from tqdm import tqdm
from copy import copy
from model import model
from utils import generate_reward


data_file = open("parsed_soc_data.json")
parsed_soc_data = json.load(data_file)

data_file = open("sanity_test_data.json")
parsed_soc_data = json.load(data_file)["test1"]

# train the model for given number of iterations and given training algo
def train(iterations, algo, epsilon):
    actual_mean = parsed_soc_data["borda_scores"]   #the actual borda score of each candidate
    num_candidates = parsed_soc_data["num_candidates"]
    num_voters = parsed_soc_data["num_voters"]
    full_voting_profile = parsed_soc_data["full_voting_profile"]
    flattened_voting_profile = parsed_soc_data["flattened_voting_profile"]

    # get the winning ballet from the borda scores of candidates
    winning_ballot = [int(i) for i in list(dict(sorted(actual_mean.items(), key=lambda item: item[1], reverse=True)).keys())]
    print("winning_ballot ", winning_ballot)


    # generate borda score dictionary for candidates and initial with 0
    reward = {}
    for cand in range(num_candidates):
        reward[cand] = 0

    agent = model(epsilon, num_candidates, num_voters)

    print("Actual mean ", actual_mean)
    print("before running mean = ", agent.mean_reward)

    # append all votes to list, create a list of dictionary for all voters having votes of each voting round
    voter_ballet_dict = {}      # dictionary containing each voter ballet list throughout the iterations

    for voter in range(num_voters):
        voter_ballet_dict[voter] = {}
        voter_ballet_dict[voter]["reward"] = {}
        for cand in range(num_candidates):
            voter_ballet_dict[voter]["reward"][cand] = 0

    for iter in tqdm(range(iterations)):
        candidate_votes = {}        # dictionary containing number votes per candidate

        # run one voting cycle where all the voters cast their vote and give the preferences
        for voter in range(num_voters):
            top_candidate = agent.pick_arm(algo, reward, voter, voter_ballet_dict[voter])
            if top_candidate in candidate_votes:
                candidate_votes[top_candidate] += 1
            else:
                candidate_votes[top_candidate] = 1

        print("winner profile ", candidate_votes)

        # get the list of winning candidates, if there are more than two winning candidates chooce one randomly (tie breaking)
        highest_vote = max(candidate_votes.values())
        winning_candidate_list = list(filter(lambda x: candidate_votes[x] == highest_vote, candidate_votes))
        print(winning_candidate_list)
        winning_candidate = random.choice(winning_candidate_list)
        print("winner ", winning_candidate)

        # get the borda score by comapring the winning candidate and the actual voter preference, update the reward for each voter
        for voter in range(num_voters):
            actual_voter_ballet = full_voting_profile[voter]
            borda_score = num_candidates - 1 - actual_voter_ballet.index(winning_candidate)
            voter_ballet_dict[voter]["reward"][winning_candidate] += borda_score
    
        print("exploit explore ", agent.exploit, agent.explore)

    for voter in range(num_voters):
        final_round_winner = max(voter_ballet_dict[voter]["reward"], key = voter_ballet_dict[voter]["reward"].get)
        print("voter {} - final_round_winner {}".format(voter, final_round_winner))

    print("borda scores of voters dictionary ", voter_ballet_dict)


train(1000, 1, 0.5)


# at the end of each iteration calcualte the borda score for each voter - the reward for the each voter will be the comaprison of the borda score calculated at the end and his preference
# sanity check - where all voters preferences are same
# 4 - 1 > 2, 3
# 3 - 3, 2, 1
# 3 - 2, 3,1 