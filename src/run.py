import json
import itertools
from tqdm import tqdm
from copy import copy
from model import model
from utils import generate_reward

data_file = open("parsed_soc_data.json")
parsed_soc_data = json.load(data_file)

# train the model for given number of iterations and given training algo
def train(iterations, algo, epsilon):
    actual_mean = parsed_soc_data["borda_scores"]   #the actual borda score of each candidate
    num_candidates = parsed_soc_data["num_candidates"]
    num_voters = parsed_soc_data["num_voters"]

    # get the winning ballet from the borda scores of candidates
    winning_ballot = [int(i) for i in list(dict(sorted(actual_mean.items(), reverse=True)).keys())]
    print("winning_ballot ", winning_ballot)

    # full_voting_profile = parsed_soc_data["full_voting_profile"]
    # flattened_voting_profile = parsed_soc_data["flattened_voting_profile"]

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
        voter_ballet_dict[voter]["ballet"] = {}
        voter_ballet_dict[voter]["reward"] = {}
        for ballet in list(itertools.permutations(range(num_candidates))):
            voter_ballet_dict[voter]["ballet"][json.dumps(ballet)] = 0
            voter_ballet_dict[voter]["reward"][json.dumps(ballet)] = 0

    for iter in tqdm(range(iterations)):
        # create a dictionary to hold the ballet order and its multiplicity
        voter_preferences = {}

        # create a dictionary of voter and their ballot for each round
        voter_ballet_iter = {}

        # run one voting cycle where all the voters cast their vote and give the preferences
        for voter in range(num_voters):
            top_candidate, voter_ballet = agent.pick_arm(algo, reward, voter, voter_ballet_dict[voter], voter_preferences, winning_ballot)
            voter_ballet = json.dumps(voter_ballet)

            voter_ballet_iter[voter] = voter_ballet

            if voter_ballet in voter_preferences:
                voter_preferences[voter_ballet] += 1
            else:
                voter_preferences[voter_ballet] = 1

            # Update the voter_ballet_dict with current ballet and it's multiplicity
            if voter_ballet in voter_ballet_dict[voter]["ballet"]:
                voter_ballet_dict[voter]["ballet"][voter_ballet] += 1
            else:
                voter_ballet_dict[voter]["ballet"][voter_ballet] = 1

            # print("voter_ballet ", voter_ballet)
            # reward = generate_reward(voter_preferences, num_candidates)

        # reward = generate_reward(voter_preferences, num_candidates)
        # print("reward ", reward)

        # update the reward of voters
        for voter, ballet in voter_ballet_iter.items():
            top_cand = json.loads(ballet)[0]
            voter_borda_score = num_candidates - 1 - winning_ballot.index(top_cand)

            # calculate the reward for each voter for a given voting ballet based on the borda scores at the end of each iteration
            voter_ballet_dict[voter]["reward"][ballet] = (voter_ballet_dict[voter]["reward"][ballet] + voter_borda_score) / voter_ballet_dict[voter]["ballet"][ballet]
            print('voter_ballet_dict[voter]["reward"][ballet] - ', voter_ballet_dict[voter]["reward"])
         
        # update the mean after every voting cycle is done
        agent.update_mean(reward)

    borda_scores = generate_reward(voter_preferences, num_candidates)
    print("borda_scores ", borda_scores)
    print("exploit explore ", agent.exploit, agent.explore)


train(1500, 1, 0.5)


# at the end of each iteration calcualte the borda score for each voter - the reward for the each voter will be the comaprison of the borda score calculated at the end and his preference
# sanity check - where all voters preferences are same
# 4 - 1 > 2, 3
# 3 - 3, 2, 1
# 3 - 2, 3,1 