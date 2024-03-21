import json
import random
import numpy as np
from model import model
from train import train_model
from data_pre_process import process_soc_data
import matplotlib.pyplot as plt
from tqdm import tqdm
from preflibtools.properties import borda_scores
from itertools import permutations, combinations


# data_file_name = {
#     "name": "sanity_test_data.json",
#     "test_case": "test3"
# }
# data_file = open(data_file_name["name"])
# parsed_soc_data = json.load(data_file)[data_file_name["test_case"]]


data_file_name = {
    "name": "parsed_soc_data",
    "test_case": ""
}
input_config = open("config.json")
input_conf = json.load(input_config)


# declare all model hyper paramters
avg_runs = 100
iterations = 100000
batch = 1000
num_voters = 5
num_candidates = 3
# voting_rule = "plurality"
voting_rule = "borda"
# voting_rule = "approval"
# voting_rule = "copeland"

approval_count_list = []

if voting_rule == "approval":
    approval_count_list = range(1, (num_candidates // 2) + 2)    # creating an array of all possible number of approval counts from 1 to num_candidates/2 + 2


# Generate a list of data for each avg_run
parsed_soc_data_list = []
for i in range(avg_runs):
    parsed_soc_data_list.append(process_soc_data(num_voters, num_candidates, True))

output_file = "results_soc_data.json"
result = {}

num_iter_arr = [i for i in range(batch, iterations + 1, batch)]


print("Voting rule ", voting_rule)

# For approval voting rule
if voting_rule == "approval":
    # Get the highest approval count possible, average it over avg_runs for the different data points generated in previous step
    actual_highest_vote_per_approval = {}       # dictionary to store average highest vote for different approval count
    for approval_count in approval_count_list:
        actual_highest_vote_per_approval[approval_count] = 0

    actual_winning_score_dict = {}
    for approval_count in approval_count_list:
        for parsed_soc_data in parsed_soc_data_list:
            for cand in range(num_candidates):
                actual_winning_score_dict[cand] = 0
            for voter in range(num_voters):
                actual_voter_ballet = parsed_soc_data["full_voting_profile"][voter]
                print("actual_voter_ballet ", actual_voter_ballet, actual_voter_ballet[:approval_count])
                for cand in actual_voter_ballet[:approval_count]:
                    actual_winning_score_dict[cand] += 1
            print(actual_winning_score_dict, "approval count ", approval_count)
            max_approval = max(actual_winning_score_dict.values())
            top_cand_list = list(filter(lambda x: actual_winning_score_dict[x] == max_approval, actual_winning_score_dict))
            winning_candidate = random.choice(top_cand_list)

            for voter in range(num_voters):
                actual_voter_ballet = parsed_soc_data["full_voting_profile"][voter]
                borda_score = num_candidates - 1 - actual_voter_ballet.index(winning_candidate)
                actual_winning_score_dict[voter] = borda_score

            actual_highest_vote_per_approval[approval_count] += sum(actual_winning_score_dict.values())

        actual_highest_vote_per_approval[approval_count] /= avg_runs
    print("actual_highest_vote_per_approval ", actual_highest_vote_per_approval)

    # Generate a list of best approval score values for test_0 as a reference point
    for approval_count in approval_count_list:
        result[approval_count] = {}
        result[approval_count]["test_0"] = {}
        result[approval_count]["test_0"]["avg_score_arr"] = [actual_highest_vote_per_approval[approval_count]]*int(iterations / batch)

    # Loop through the different approval count list and different test cases to generate results
    for approval_count in approval_count_list:
        print("approval count - ", approval_count)
        for key in input_conf.keys():
            print("epsilon - ", input_conf[key]["epsilon"], " decay factor - ", input_conf[key]["epsilon_decay"])
            print("key - ", key)

            agent = model(input_conf[key]["epsilon"], num_candidates, num_voters, approval_count)
            result[approval_count][key] = {}
            avg_score_arr = [0]*(iterations//batch)
            score_arr = []
            for i in tqdm(range(avg_runs)):
                train = train_model(agent, iterations, batch, voting_rule, parsed_soc_data_list[i], input_conf[key]["epsilon"], input_conf[key]["grad_epsilon"], input_conf[key]["epsilon_final"], input_conf[key]["epsilon_decay"], approval_count)   # for each iteration returns a dictionary containing voter and there fnal reward i.e borda score of top candidate
                score, welfare_dict_list = train.train(1, input_conf[key]["single_iterative_voting"])
                for j in range(iterations//batch):
                    avg_score_arr[j] += score[j]

            for i in range(iterations//batch):
                avg_score_arr[i] /= avg_runs
            result[approval_count][key]["avg_score_arr"] = avg_score_arr
            result[approval_count][key]["welfare_dict_list"] = welfare_dict_list

        plot = plt.figure()
        for key in result[approval_count].keys():
            print(result[approval_count][key]["avg_score_arr"][0], result[approval_count][key]["avg_score_arr"][-1])
            plt.plot(num_iter_arr, result[approval_count][key]["avg_score_arr"], label=key)

        plt.legend(loc='upper right')
        plt.xlabel("Number of iterations")
        plt.ylabel("Avg borda Score with approval count - " + str(approval_count))
        # plt.ylim(0, actual_highest_vote_per_approval[approval_count] + 1)
        plt.show()
        plt.savefig('results/approval_count_voter_' + str(num_voters) + '_cand_' + str(num_candidates) + \
                    "_iter_" + str(iterations) + "_avg_" + str(avg_runs) + '.png')


# For plurality voting rule
elif voting_rule == "plurality":
    
    # calculate borda score based on the true preferences of the voters over all the generated data
    actual_winning_score_dict = {}
    avg_highest_borda_score = 0

    for parsed_soc_data in parsed_soc_data_list:
        voter_ballot_iter = {}        # dictionary containing voter and their corresponding ballot
        for voter_i in range(num_voters):
            cand = parsed_soc_data["full_voting_profile"][voter_i][0]
            voter_ballot_iter[voter_i] = cand

        candidate_votes = {}
        for top_candidate in voter_ballot_iter.values():
            if top_candidate in candidate_votes:
                candidate_votes[top_candidate] += 1
            else:
                candidate_votes[top_candidate] = 1

        highest_vote = max(candidate_votes.values())
        winning_candidate_list = list(filter(lambda x: candidate_votes[x] == highest_vote, candidate_votes))
        winning_candidate = random.choice(winning_candidate_list)

        for voter in range(num_voters):
            actual_voter_ballet = parsed_soc_data["full_voting_profile"][voter]
            borda_score = num_candidates - 1 - actual_voter_ballet.index(winning_candidate)
            actual_winning_score_dict[voter] = borda_score

        avg_highest_borda_score += sum(actual_winning_score_dict.values())
    
    avg_highest_borda_score /= len(parsed_soc_data_list) 
    print("average highest borda score: ", avg_highest_borda_score)

    result["test_0"] = {}
    result["test_0"]["avg_score_arr"] = [avg_highest_borda_score]*int(iterations / batch)

    for key in input_conf.keys():

        print("key - ", key)
        print("epsilon - ", input_conf[key]["epsilon"], " decay factor - ", input_conf[key]["epsilon_decay"])

        agent = model(input_conf[key]["epsilon"], num_candidates, num_voters, approval_count=0)
        result[key] = {}
        avg_score_arr = [0]*(iterations//batch)
        score_arr = []
        for i in tqdm(range(avg_runs)):
            train = train_model(agent, iterations, batch, voting_rule, parsed_soc_data_list[i], input_conf[key]["epsilon"], input_conf[key]["grad_epsilon"], input_conf[key]["epsilon_final"], input_conf[key]["epsilon_decay"])   # for each iteration returns a dictionary containing voter and there fnal reward i.e borda score of top candidate
            score, welfare_dict_list = train.train(1, input_conf[key]["single_iterative_voting"])
            for j in range(iterations//batch):
                avg_score_arr[j] += score[j]

        for i in range(iterations//batch):
            avg_score_arr[i] /= avg_runs

        result[key]["avg_score_arr"] = avg_score_arr
        result[key]["welfare_dict_list"] = welfare_dict_list

    plot = plt.figure()
    for key in result.keys():
        print(result[key]["avg_score_arr"][0], result[key]["avg_score_arr"][-1])
        plt.plot(num_iter_arr, result[key]["avg_score_arr"], label=key)
        # plt.text("epsilon - {}, epsilon decay - {}".format(input_conf[key]["epsilon"], input_conf[key]["epsilon_decay"]))

    plt.legend(loc='upper right')
    plt.xlabel("Number of iterations")
    plt.ylabel("Avg Borda Score with plurality")
    # plt.ylim(0, actual_highest_vote_per_approval + 1)
    plt.show()
    plt.savefig('results/plurality_voter_' + str(num_voters) + '_cand_' + str(num_candidates) + \
                    "_iter_" + str(iterations) + "_avg_" + str(avg_runs) + '.png')


elif voting_rule == "borda":
    # calculate borda score based on the true preferences of the voters over all the generated data
    actual_winning_score_dict = {}
    avg_highest_borda_welfare = 0

    for parsed_soc_data in parsed_soc_data_list:
        print(parsed_soc_data["full_voting_profile"])
        voter_ballot_iter = {}        # dictionary containing voter and their corresponding ballot
        for voter_i in range(num_voters):
            voter_ballot_iter[voter_i] = parsed_soc_data["full_voting_profile"][voter_i]

        candidate_votes = {}
        for ballot in voter_ballot_iter.values():
            for cand in ballot:
                if cand in candidate_votes:
                    candidate_votes[cand] += num_candidates - 1 - ballot.index(cand)
                else:
                    candidate_votes[cand] =  num_candidates - 1 - ballot.index(cand)
        print(candidate_votes)
        highest_vote = max(candidate_votes.values())
        winning_candidate_list = list(filter(lambda x: candidate_votes[x] == highest_vote, candidate_votes))
        winning_candidate = random.choice(winning_candidate_list)

        for voter in range(num_voters):
            actual_voter_ballet = parsed_soc_data["full_voting_profile"][voter]
            borda_score = num_candidates - 1 - actual_voter_ballet.index(winning_candidate)
            actual_winning_score_dict[voter] = borda_score
        avg_highest_borda_welfare += sum(actual_winning_score_dict.values())
    
    avg_highest_borda_welfare /= len(parsed_soc_data_list) 
    print("average highest welfare of voters: ", avg_highest_borda_welfare)

    result["test_0"] = {}
    result["test_0"]["avg_score_arr"] = [avg_highest_borda_welfare]*int(iterations / batch)

    for key in input_conf.keys():
        print("key - ", key)
        print("epsilon - ", input_conf[key]["epsilon"], " decay factor - ", input_conf[key]["epsilon_decay"])

        agent = model(input_conf[key]["epsilon"], num_candidates, num_voters, approval_count=0)
        result[key] = {}
        avg_score_arr = [0]*(iterations//batch)
        score_arr = []
        for i in tqdm(range(avg_runs)):
            train = train_model(agent, iterations, batch, voting_rule, parsed_soc_data_list[i], input_conf[key]["epsilon"], input_conf[key]["grad_epsilon"], input_conf[key]["epsilon_final"], input_conf[key]["epsilon_decay"])   # for each iteration returns a dictionary containing voter and there fnal reward i.e borda score of top candidate
            score, welfare_dict_list = train.train(1, input_conf[key]["single_iterative_voting"])
            for j in range(iterations//batch):
                avg_score_arr[j] += score[j]

        for i in range(iterations//batch):
            avg_score_arr[i] /= avg_runs

        result[key]["avg_score_arr"] = avg_score_arr
        result[key]["welfare_dict_list"] = welfare_dict_list

    plot = plt.figure()
    for key in result.keys():
        print(result[key]["avg_score_arr"][0], result[key]["avg_score_arr"][-1])
        plt.plot(num_iter_arr, result[key]["avg_score_arr"], label=key)
        # plt.text("epsilon - {}, epsilon decay - {}".format(input_conf[key]["epsilon"], input_conf[key]["epsilon_decay"]))


    plt.legend(loc='upper right')
    plt.xlabel("Number of iterations")
    plt.ylabel("Avg Borda Score with Borda")
    # plt.ylim(0, actual_highest_vote_per_approval + 1)
    plt.show()
    plt.savefig('results/borda_voter_' + str(num_voters) + '_cand_' + str(num_candidates) + \
                    "_iter_" + str(iterations) + "_avg_" + str(avg_runs) + '.png')


elif voting_rule == "copeland":
    actual_winning_score_dict = {}
    avg_highest_copeland_welfare = 0

    for parsed_soc_data in parsed_soc_data_list:
        # print(parsed_soc_data["full_voting_profile"])
        voter_ballot_iter = {}        # dictionary containing voter and their corresponding ballot
        for voter_i in range(num_voters):
            voter_ballot_iter[voter_i] = parsed_soc_data["full_voting_profile"][voter_i]

        pair_wise_combinations = combinations(range(num_candidates), 2)
        pair_wise_winner = {}
        for cand in range(num_candidates):
            pair_wise_winner[cand] = 0

        for pair in pair_wise_combinations:
            for ballot in voter_ballot_iter.values():
                if ballot.index(pair[0]) > ballot.index(pair[1]):
                    pair_wise_winner[pair[1]] += 1
                else:
                    pair_wise_winner[pair[0]] += 1

        # print("pair_wise_winner ", pair_wise_winner)
        highest_copeland_score = max(pair_wise_winner.values())
        winning_candidate_list = list(filter(lambda x: pair_wise_winner[x] == highest_copeland_score, pair_wise_winner))
        winning_candidate = random.choice(winning_candidate_list)

        for voter in range(num_voters):
            actual_voter_ballet = parsed_soc_data["full_voting_profile"][voter]
            borda_score = num_candidates - 1 - actual_voter_ballet.index(winning_candidate)
            actual_winning_score_dict[voter] = borda_score
        avg_highest_copeland_welfare += sum(actual_winning_score_dict.values())
    
    avg_highest_copeland_welfare /= len(parsed_soc_data_list) 
    print("average highest copeland score: ", avg_highest_copeland_welfare)

    result["test_0"] = {}
    result["test_0"]["avg_score_arr"] = [avg_highest_copeland_welfare]*int(iterations / batch)

    for key in input_conf.keys():
        print("key - ", key)
        print("epsilon - ", input_conf[key]["epsilon"], " decay factor - ", input_conf[key]["epsilon_decay"])

        agent = model(input_conf[key]["epsilon"], num_candidates, num_voters, approval_count=0)
        result[key] = {}
        avg_score_arr = [0]*(iterations//batch)
        score_arr = []
        for i in tqdm(range(avg_runs)):
            train = train_model(agent, iterations, batch, voting_rule, parsed_soc_data_list[i], input_conf[key]["epsilon"], input_conf[key]["grad_epsilon"], input_conf[key]["epsilon_final"], input_conf[key]["epsilon_decay"])   # for each iteration returns a dictionary containing voter and there fnal reward i.e copeland score of top candidate
            score, welfare_dict_list = train.train(1, input_conf[key]["single_iterative_voting"])
            for j in range(iterations//batch):
                avg_score_arr[j] += score[j]

        for i in range(iterations//batch):
            avg_score_arr[i] /= avg_runs

        result[key]["avg_score_arr"] = avg_score_arr
        result[key]["welfare_dict_list"] = welfare_dict_list

    plot = plt.figure()
    for key in result.keys():
        print(result[key]["avg_score_arr"][0], result[key]["avg_score_arr"][-1])
        plt.plot(num_iter_arr, result[key]["avg_score_arr"], label=key)
        # plt.text("epsilon - {}, epsilon decay - {}".format(input_conf[key]["epsilon"], input_conf[key]["epsilon_decay"]))


    plt.legend(loc='upper right')
    plt.xlabel("Number of iterations")
    plt.ylabel("Avg borda Score with copeland voting rule")
    # plt.ylim(0, actual_highest_vote_per_approval + 1)
    plt.show()
    plt.savefig('results/copeland_voter_' + str(num_voters) + '_cand_' + str(num_candidates) + \
                    "_iter_" + str(iterations) + "_avg_" + str(avg_runs) + '.png')


with open(output_file, "w") as f:
    json.dump(result, f)

