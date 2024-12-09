import os
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
committee_size_list = [3]  # ( k will be the committee size)
num_voters = 10
num_candidates = 5
approval_count = 2

tie_breaking_rule = "rand"
# tie_breaking_rule = "dict"


# voting_rule = "stv"
# voting_rule = "pav"
# voting_rule = "bloc"
# voting_rule = "monroe"
# voting_rule = "chamberlin_courant"
# voting_rule = "plurality"
# voting_rule = "anti_plurality"
voting_rule = "borda"
# voting_rule = "borda_top_cand"
# voting_rule = "approval"
# voting_rule = "copeland"


avg_runs = 50
iterations = 50000
batch = 500

# avg_runs = 1
# iterations = 10
# batch = 1

# create dictionary to save the monotonicity results of different test configs
monotonicity_check = {}
monotonicity_check['count'] = {}
for i in range(avg_runs):
    monotonicity_check['run_{}'.format(i)] = {}
    for key in input_conf.keys():
        monotonicity_check['run_{}'.format(i)][key] = {}


# if voting_rule == "approval":
#     approval_count_list = list(range(1, (num_candidates // 2) + 2))    # creating an array of all possible number of approval counts from 1 to num_candidates/2 + 2


# Generate a list of data for each avg_run
parsed_soc_data_list = []
for i in range(avg_runs):
    parsed_soc_data_list.append(process_soc_data(num_voters, num_candidates, True))


# preference profiles, rewards and winning committes based on voters true preferences
test_details = {}


# get the parent directry and write create subdirectries for different runs
curr_dir = os.path.dirname(os.getcwd())

# create a folder for given voting rule if the folder doest exsit already
os.makedirs(curr_dir + "/numerical_results/" + voting_rule, exist_ok=True)
os.makedirs(curr_dir + "/graph_results/" + voting_rule, exist_ok=True)


# save config setting and iteration details in result dictionary
result = {}
result["graph_coords"] = {}


num_iter_arr = [i for i in range(batch, iterations + 1, batch)]

print("Voting rule ", voting_rule, " voting_setting ", voting_setting, "tie_breaking_rule ", tie_breaking_rule)

for committee_size in committee_size_list:
    if voting_rule == "approval":
        output_file = 'setting_{}_{}_tie_breaking_{}_approval_count_{}_voter_'.format(voting_setting, voting_rule, tie_breaking_rule, approval_count) + str(num_voters) + '_cand_' + str(num_candidates) + \
                    "_committee_size_" + str(committee_size) + "_iter_" + str(iterations) + "_avg_" + str(avg_runs) + '.json'
    else:
        output_file = 'setting_{}_{}_tie_breaking_{}_voter_'.format(voting_setting, voting_rule, tie_breaking_rule) + str(num_voters) + '_cand_' + str(num_candidates) + \
                    "_committee_size_" + str(committee_size) + "_iter_" + str(iterations) + "_avg_" + str(avg_runs) + '.json'

    # file path for numerical results of the experiment, store all the data collected during the experiment in this file
    output_path = os.path.join(curr_dir + "/numerical_results/" + voting_rule + "/", output_file)


    for key in input_conf.keys():
        print("key - ", key)
        print("epsilon - ", input_conf[key]["epsilon"], " decay factor - ", input_conf[key]["epsilon_decay"])

        agent = model(input_conf[key]["epsilon"], num_candidates, committee_size, num_voters, approval_count)
        result["graph_coords"][key] = {}
        avg_score_arr = [0]*(iterations//batch)
        # avg_num_ties = [0]*(iterations//batch)
        score_arr = []
        run_details = {}
        for i in tqdm(range(avg_runs)):
            train = train_model(agent, iterations, batch, committee_size, voting_rule, tie_breaking_rule, voting_setting, parsed_soc_data_list[i], input_conf[key]["epsilon"], input_conf[key]["grad_epsilon"], input_conf[key]["epsilon_final"], input_conf[key]["epsilon_decay"], approval_count)   # for each iteration returns a dictionary containing voter and there fnal reward i.e borda score of top candidate
            score, welfare_dict_list, winners_list, voter_ballot_iter_list, num_ties_list = train.train(1, input_conf[key]["single_iterative_voting"])
            
            run_details['run_{}'.format(i)] = {}
            run_details['run_{}'.format(i)]["true_preference_values"] = {
                "profile" : parsed_soc_data_list[i]["full_voting_profile"],
                "reward" : score[0],
                "winner" : winners_list[0]
            }

            run_details['run_{}'.format(i)]["predicted_prefrence_values"] = {
                "profile" : voter_ballot_iter_list,
                "rewards" : score,
                "winners" : winners_list,
                # "num_ties" : num_ties_list
            }

            monotonicity_check['run_{}'.format(i)][key]['committee_size_{}'.format(committee_size)] = winners_list[-1]

            for j in range(0, iterations, batch):
                avg_score_arr[j//batch] += score[j]
                # avg_num_ties[j//batch] += num_ties_list[j]

        test_details[key] = run_details

        for i in range(iterations//batch):
            avg_score_arr[i] /= avg_runs
            # avg_num_ties[i] /= avg_runs
        
        result["graph_coords"][key]["avg_score_arr"] = avg_score_arr
        # result["graph_coords"][key]["avg_num_ties"] = avg_num_ties
        # result["graph_coords"][key]["welfare_dict_list"] = welfare_dict_list

    result["graph_coords"]["test_0"] = {}
    result["graph_coords"]["test_0"]["avg_score_arr"] = [avg_score_arr[0]]*(iterations//batch)
    result["graph_coords"]["test_0"]["avg_num_ties"] = [0]*(iterations//batch)


    # graphs for showing avg borda score for given voting setting 
    plot = plt.figure()
    for key in result["graph_coords"].keys():
        print(result["graph_coords"][key]["avg_score_arr"][0], result["graph_coords"][key]["avg_score_arr"][-1])
        plt.plot(num_iter_arr, result["graph_coords"][key]["avg_score_arr"], label=key)
        # plt.text("epsilon - {}, epsilon decay - {}".format(input_conf[key]["epsilon"], input_conf[key]["epsilon_decay"]))

    plt.legend(loc='upper right')
    plt.xlabel("Number of iterations")
    plt.ylabel("Avg Score with {}".format(voting_rule))
    # plt.ylim(0, actual_highest_vote_per_approval + 1)
    plt.show()
    if voting_rule == "approval":
        graph_file = 'avg_score_setting_{}_{}_tie_breaking_{}_approval_count_{}_voter_'.format(voting_setting, voting_rule, tie_breaking_rule, approval_count) + str(num_voters) + '_cand_' + str(num_candidates) + \
                    "_committee_size_" + str(committee_size) + "_iter_" + str(iterations) + "_avg_" + str(avg_runs) + '.png'
    else:
        graph_file = 'avg_score_setting_{}_{}_tie_breaking_{}_voter_'.format(voting_setting, voting_rule, tie_breaking_rule) + str(num_voters) + '_cand_' + str(num_candidates) + \
                    "_committee_size_" + str(committee_size) + "_iter_" + str(iterations) + "_avg_" + str(avg_runs) + '.png'
    graph_path = os.path.join(curr_dir + "/graph_results/" + voting_rule + "/", graph_file)

    plt.savefig(graph_path)


    # # PLots for Avg number of ties using different tie breaking rules
    # plot = plt.figure()
    # for key in result["graph_coords"].keys():
    #     plt.plot(num_iter_arr, result["graph_coords"][key]["avg_num_ties"], label="{}_ties".format(key))

    # plt.legend(loc='upper right')
    # plt.xlabel("Number of iterations")
    # plt.ylabel("Avg Number of ties with {}".format(voting_rule))
    # # plt.ylim(0, actual_highest_vote_per_approval + 1)
    # plt.show()
    # if voting_rule == "approval":
    #     graph_file = 'tie_breaking_{}_setting_{}_{}_approval_count_{}_voter_'.format(tie_breaking_rule, voting_setting, voting_rule, approval_count) + str(num_voters) + '_cand_' + str(num_candidates) + \
    #                 "_committee_size_" + str(committee_size) + "_iter_" + str(iterations) + "_avg_" + str(avg_runs) + '.png'
    # else:
    #     graph_file = 'tie_breaking_{}_setting_{}_{}_voter_'.format(tie_breaking_rule, voting_setting, voting_rule) + str(num_voters) + '_cand_' + str(num_candidates) + \
    #                 "_committee_size_" + str(committee_size) + "_iter_" + str(iterations) + "_avg_" + str(avg_runs) + '.png'
    # graph_path = os.path.join(curr_dir + "/graph_results/" + voting_rule + "/", graph_file)

    # plt.savefig(graph_path)


    result["config"] = input_conf
    result["run_setting"] = {
        "voting_setting" : voting_setting,  # 0 - single winner setting, 1 - committee voting
        "committee_size" : committee_size,  # ( k will be the committee size)
        "num_voters" : num_voters,
        "num_candidates" : num_candidates,
        "avg_runs" : avg_runs,
        "iterations" : iterations,
        "batch" : batch,
        "voting_rule" : voting_rule,
        "tie_breaking_rule": tie_breaking_rule,
        "approval_count" : approval_count,
        "test_details" : test_details
    }

    with open(output_path, "w+") as f:
        json.dump(result, f, default=int)


print(voting_rule)
print(output_file)

"""
# Monotonicity check
if len(committee_size_list) > 1:
    # monotonicity check for avg runs over different test cases
    monotonicity_count = {}
    for test in input_conf.keys():
        monotonicity_count[test] = avg_runs

    for run in range(avg_runs):
        for test in input_conf.keys():
            winner_list = list(monotonicity_check['run_{}'.format(run)][test].values())
            if len([winner for winner in winner_list[1] if winner not in winner_list[0]]) > 1:
                monotonicity_count[test] -= 1

    monotonicity_check['count'] = monotonicity_count


    print(monotonicity_check)
    output_file = 'monotonicity_check_setting_{}_{}_voter_'.format(voting_setting, voting_rule) + str(num_voters) + '_cand_' + str(num_candidates) + \
                    "_committee_size_" + '_'.join(str(size) for size in committee_size_list) + "_iter_" + str(iterations) + "_avg_" + str(avg_runs) + '.json'
    output_path = os.path.join(curr_dir + "/numerical_results/" + voting_rule + "/", output_file)

    with open(output_path, "w+") as f:
        json.dump(monotonicity_check, f, default=int)

"""
