import json
import random
import itertools
from tqdm import tqdm
from copy import copy
from model import model
from utils import generate_reward


# data_file = open("parsed_soc_data.json")
# parsed_soc_data = json.load(data_file)

# data_file = open("sanity_test_data.json")
# parsed_soc_data = json.load(data_file)["test2"]

# train the model for given number of iterations and given training algo
def train(iterations, batch, algo, epsilon, parsed_soc_data):
    actual_mean = parsed_soc_data["borda_scores"]   #the actual borda score of each candidate
    num_candidates = parsed_soc_data["num_candidates"]
    num_voters = parsed_soc_data["num_voters"]
    full_voting_profile = parsed_soc_data["full_voting_profile"]
    flattened_voting_profile = parsed_soc_data["flattened_voting_profile"]

    borda_scores_arr = []   # list of borda scores at given n iterations


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

    # print("Actual mean ", actual_mean)
    # print("before running mean = ", agent.mean_reward)

    # append all votes to list, create a list of dictionary for all voters having votes of each voting round
    voter_ballet_dict = {}      # dictionary containing each voter ballet list throughout the iterations

    for voter in range(num_voters):
        voter_ballet_dict[voter] = {}
        voter_ballet_dict[voter]["reward"] = {}
        voter_ballet_dict[voter]["count"] = {}
        for cand in range(num_candidates):
            voter_ballet_dict[voter]["reward"][cand] = 0
            voter_ballet_dict[voter]["count"][cand] = 0

    voter_top_cand_iter = {}        # dictionary containing voter and their corresponding top candidate
    
    n_iter_reward = {}
    for voter in range(num_voters):
        n_iter_reward[voter] = 0

    for iter in range(iterations):
        candidate_votes = {}        # dictionary containing number votes per candidate

        # run one voting cycle where all the voters cast their vote and give the preferences
        # for voter in range(num_voters):
        #     voter_top_cand_iter[voter] = voter_top_cand_iter[voter]
        #     top_candidate = agent.pick_arm(algo, reward, voter, voter_ballet_dict[voter])
        #     voter_top_cand_iter[voter] = top_candidate
        #     voter_ballet_dict[voter]["count"][cand] += 1

        # In each iteration pick one voter randomly and they will use the pick arm method to vote
        manupilating_voter = random.choice([i for i in range(num_voters)])
        top_candidate = agent.pick_arm(algo, reward, manupilating_voter, voter_ballet_dict[manupilating_voter])
        voter_top_cand_iter[manupilating_voter] = top_candidate
        voter_ballet_dict[manupilating_voter]["count"][top_candidate] += 1

        voter_list = [i for i in range(num_voters)]
        voter_list.remove(manupilating_voter)
        for voter in voter_list:
            cand = full_voting_profile[voter][0]  # initialy asign this dictionary with true preferences of voters
            voter_top_cand_iter[voter] = cand
            voter_ballet_dict[voter]["count"][cand] += 1


        # update the candidate votes dictionary
        for top_candidate in voter_top_cand_iter.values():
            if top_candidate in candidate_votes:
                candidate_votes[top_candidate] += 1
            else:
                candidate_votes[top_candidate] = 1

        # get the borda score by comapring the winning candidate and the actual voter preference, update the reward for each voter for their voted candidate
        for voter in range(num_voters):
            actual_voter_ballet = full_voting_profile[voter]
            top_cand = voter_top_cand_iter[voter]
            borda_score = num_candidates - 1 - actual_voter_ballet.index(top_cand)
            voter_ballet_dict[voter]["reward"][top_cand] = (voter_ballet_dict[voter]["reward"][top_cand] + borda_score)

        if iter % batch == 0:
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
            print("winning_candidate ", winning_candidate)
            print("final_round_reward_cand ", borda_scores_arr)

        # print("winner profile ", candidate_votes)

        # get the list of winning candidates, if there are more than two winning candidates chooce one randomly (tie breaking)
        # highest_vote = max(candidate_votes.values())
        # winning_candidate_list = list(filter(lambda x: candidate_votes[x] == highest_vote, candidate_votes))
        # winning_candidate = random.choice(winning_candidate_list)

        # if iter % batch == 0:
        #     print("candidate_votes ", candidate_votes)
        #     print("borda scores of voters dictionary ", voter_ballet_dict)

        #     # create a preference profile based on the rewards the voters got in last round
        #     # generated_preference_profile = {}
        #     # for voter in range(num_voters):
        #     #     generated_preference_profile[voter] = sorted(voter_ballet_dict[voter]["reward"], key=voter_ballet_dict[voter]["reward"].get, reverse=True)
            
        #     # print("generated_preference_profile ", generated_preference_profile)
        
            
        #     winning_borda_score = 0
        #     for voter in range(num_voters):
        #         actual_voter_ballet = full_voting_profile[voter]
        #         borda_score = num_candidates - 1 - actual_voter_ballet.index(winning_candidate)
        #         winning_borda_score += borda_score
            
        #     borda_scores_arr.append(winning_borda_score)
        #     print("winning_candidate ", winning_borda_score)
        #     print("final_round_reward_cand ", borda_scores_arr)



        # for voter in range(num_voters):
        #     actual_voter_ballet = full_voting_profile[voter]
        #     borda_score = num_candidates - 1 - actual_voter_ballet.index(winning_candidate)
        #     voter_ballet_dict[voter]["reward"][voter_top_cand_iter[voter]] += (borda_score / voter_ballet_dict[voter]["count"][cand])
    
        # print("exploit explore ", agent.exploit, agent.explore)

    # final_round_candidate_votes = {}    # dictionary of candidate and number voters chose them in final round
    # for voter in range(num_voters):
    #     final_round_winner = max(voter_ballet_dict[voter]["reward"], key = voter_ballet_dict[voter]["reward"].get)
    #     if final_round_winner in final_round_candidate_votes:
    #         final_round_candidate_votes[final_round_winner] += 1
    #     else:
    #         final_round_candidate_votes[final_round_winner] = 1
    #     # print("voter {} - final_round_winner {}".format(voter, final_round_winner))

    # generated_borda_score_dict = {}
    # for cand in range(num_candidates):
    #     winning_borda_score = 0
    #     for voter in range(num_voters):
    #         generated_voter_ballet = generated_preference_profile[voter]
    #         borda_score = num_candidates - 1 - generated_voter_ballet.index(cand)
    #         winning_borda_score += borda_score
    #     generated_borda_score_dict[cand] = winning_borda_score

    # final_round_reward_cand = {}
    # for cand in range(num_candidates):
    #     final_round_reward_cand[cand] = 0

    # for cand in range(num_candidates):
    #     score = 0
    #     for voter in range(num_voters):
    #         score += voter_ballet_dict[voter]["reward"][cand]
    #     final_round_reward_cand[cand] = score/iter

    return borda_scores_arr
    
    # get the average borda score of the winning candidates from all voters
    winning_borda_score_dict = {}    # contains voter and there top candidates avg borda acore
    # for voter in range(num_voters):
    #     winning_borda_score_dict[voter] = 0

    for voter in range(num_voters):
        # if voter in winning_borda_score_dict:
        winning_borda_score_dict[voter] = voter_ballet_dict[voter]["reward"][voter_top_candidates[voter]]

    # winning_borda_score_dict = voter_ballet_dict[voter]["reward"][actual_winning_candidate]

    # winning_borda_score_dict /= num_voters
    # print("winning_borda_score_dict: ", winning_borda_score_dict)
    
    return winning_borda_score_dict


# train(10000, 1, 0.5, parsed_soc_data)


# at the end of each iteration calcualte the borda score for each voter - the reward for the each voter will be the comaprison of the borda score calculated at the end and his preference
# sanity check - where all voters preferences are same
# 4 - 1 > 2, 3
# 3 - 3, 2, 1
# 3 - 2, 3,1 


#TODO: write sudo code