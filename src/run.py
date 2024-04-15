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
voting_setting = 1  # 0 - single winner setting, 1 - committee voting
committee_size = 5  # ( k will be the committee size)

num_voters = 9
num_candidates = 7
# voting_rule = "plurality"
# voting_rule = "borda"
voting_rule = "approval"
# voting_rule = "copeland"


avg_runs = 50
iterations = 100000
batch = 1000
approval_count_list = []

if voting_rule == "approval":
    approval_count_list = list(range(1, (num_candidates // 2) + 2))    # creating an array of all possible number of approval counts from 1 to num_candidates/2 + 2


# Generate a list of data for each avg_run
parsed_soc_data_list = []
for i in range(avg_runs):
    parsed_soc_data_list.append(process_soc_data(num_voters, num_candidates, True))

output_file = "results_soc_data.json"
result = {}

num_iter_arr = [i for i in range(batch, iterations + 1, batch)]

print("Voting rule ", voting_rule, " voting_setting ", voting_setting)


"""Single winner setting"""

if voting_rule == "approval":
    # Loop through the different approval count list and different test cases to generate results
    for approval_count in approval_count_list:
        result[approval_count] = {}
        result[approval_count]["test_0"] = {}
        avg_score_arr = []
        print("approval count - ", approval_count)

        for key in input_conf.keys():
            print("epsilon - ", input_conf[key]["epsilon"], " decay factor - ", input_conf[key]["epsilon_decay"])
            print("key - ", key)

            agent = model(input_conf[key]["epsilon"], num_candidates, committee_size, num_voters, approval_count)
            result[approval_count][key] = {}
            avg_score_arr = [0]*(iterations//batch)
            score_arr = []
            for i in tqdm(range(avg_runs)):
                train = train_model(agent, iterations, batch, committee_size, voting_rule, voting_setting, parsed_soc_data_list[i], input_conf[key]["epsilon"], input_conf[key]["grad_epsilon"], input_conf[key]["epsilon_final"], input_conf[key]["epsilon_decay"], approval_count)   # for each iteration returns a dictionary containing voter and there fnal reward i.e borda score of top candidate
                score, welfare_dict_list = train.train(1, input_conf[key]["single_iterative_voting"])
                for j in range(iterations//batch):
                    avg_score_arr[j] += score[j]

            for i in range(iterations//batch):
                avg_score_arr[i] /= avg_runs
            result[approval_count][key]["avg_score_arr"] = avg_score_arr
            result[approval_count][key]["welfare_dict_list"] = welfare_dict_list

        plot = plt.figure()
        result[approval_count]["test_0"]["avg_score_arr"] = [avg_score_arr[0]]*(iterations//batch)

        for key in result[approval_count].keys():
            print(result[approval_count][key]["avg_score_arr"][0], result[approval_count][key]["avg_score_arr"][-1])
            plt.plot(num_iter_arr, result[approval_count][key]["avg_score_arr"], label=key)

        plt.legend(loc='upper right')
        plt.xlabel("Number of iterations")
        plt.ylabel("Avg borda Score with approval count - " + str(approval_count))
        # plt.ylim(0, actual_highest_vote_per_approval[approval_count] + 1)
        plt.show()
        plt.savefig('results/setting_{}_approval_count_{}_committee_size_{}_voter_{}_cand_{}_iter_{}_avg_{}.png'\
                    .format(voting_setting, approval_count, committee_size, num_voters, num_candidates, iterations, avg_runs))

else:
    for key in input_conf.keys():
        print("key - ", key)
        print("epsilon - ", input_conf[key]["epsilon"], " decay factor - ", input_conf[key]["epsilon_decay"])

        agent = model(input_conf[key]["epsilon"], num_candidates, committee_size, num_voters, approval_count=0)
        result[key] = {}
        avg_score_arr = [0]*(iterations//batch)
        score_arr = []
        for i in tqdm(range(avg_runs)):
            train = train_model(agent, iterations, batch, committee_size, voting_rule, voting_setting, parsed_soc_data_list[i], input_conf[key]["epsilon"], input_conf[key]["grad_epsilon"], input_conf[key]["epsilon_final"], input_conf[key]["epsilon_decay"])   # for each iteration returns a dictionary containing voter and there fnal reward i.e borda score of top candidate
            score, welfare_dict_list = train.train(1, input_conf[key]["single_iterative_voting"])
            for j in range(iterations//batch):
                avg_score_arr[j] += score[j]

        for i in range(iterations//batch):
            avg_score_arr[i] /= avg_runs

        result[key]["avg_score_arr"] = avg_score_arr
        result[key]["welfare_dict_list"] = welfare_dict_list

    result["test_0"] = {}
    result["test_0"]["avg_score_arr"] = [avg_score_arr[0]]*(iterations//batch)

    plot = plt.figure()
    for key in result.keys():
        print(result[key]["avg_score_arr"][0], result[key]["avg_score_arr"][-1])
        plt.plot(num_iter_arr, result[key]["avg_score_arr"], label=key)
        # plt.text("epsilon - {}, epsilon decay - {}".format(input_conf[key]["epsilon"], input_conf[key]["epsilon_decay"]))

    plt.legend(loc='upper right')
    plt.xlabel("Number of iterations")
    plt.ylabel("Avg Borda Score with {}".format(voting_rule))
    # plt.ylim(0, actual_highest_vote_per_approval + 1)
    plt.show()
    plt.savefig('results/setting_{}_{}_voter_'.format(voting_setting, voting_rule) + str(num_voters) + '_cand_' + str(num_candidates) + \
                    "_iter_" + str(iterations) + "_avg_" + str(avg_runs) + '.png')

with open(output_file, "w") as f:
    json.dump(result, f)

"""End of Single winner setting"""


"""Committe voting setting"""



"""End of Committe voting setting"""
