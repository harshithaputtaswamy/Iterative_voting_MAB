from data_pre_process import *
from train import *
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
borda_scores_arr = []
num_iter_arr = []


# import matplotlib
# matplotlib.use("QtAgg")


# data_file_name = {
#     "name": "sanity_test_data.json",
#     "test_case": "test3"
# }
# data_file = open(data_file_name["name"])
# parsed_soc_data = json.load(data_file)[data_file_name["test_case"]]


data_file_name = {
    "name": "parsed_soc_data.json",
    "test_case": ""
}
# data_file = open(data_file_name["name"])
# parsed_soc_data = json.load(data_file)


# approval_count_value = 
num_voters = 9
num_candidates = 7
approval_count_list = range(1, (num_candidates // 2) + 2)    # creating an array of all possible number of approval counts from 1 to num_candidates/2 + 2
# approval_count_list = [1]
print(approval_count_list)


# calculate borda score for this all candidate to see who the actual winner would be
# for cand in range(num_candidates):
#     winning_borda_score = 0
#     for voter in range(num_voters):
#         actual_voter_ballet = full_voting_profile[voter]
#         borda_score = num_candidates - 1 - actual_voter_ballet.index(cand)
#         winning_borda_score += borda_score
#     actual_winning_score_dict[cand] = winning_borda_score

# print("actual winning candidate borda score: {}", actual_winning_score_dict)

# calculate actual approval score of candidates
# approval_dict = {}
# for cand in range(num_candidates):
#     approval_dict[cand] = 0
    

# for i in range(50, 500, 50):
#     avg_borda_score = 0
#     # for j in range(i):
#     #     avg_borda_score += train(1000, 1, 0.5, parsed_soc_data)   # for each iteration returns a dictionary containing voter and there fnal reward i.e borda score of top candidate

input_file = open("config.json")
input_conf = json.load(input_file)

avg_runs = 100
iterations = 1000000
batch = 500

# Generate a list of data for each avg_run
parsed_soc_data_list = []
for i in range(avg_runs):
    parsed_soc_data = process_soc_data(num_voters, num_candidates, True)
 
    flattened_voting_profile = parsed_soc_data["flattened_voting_profile"]
    full_voting_profile = []
    for ballet, num in flattened_voting_profile.items():
        for i in range(num):
            full_voting_profile.append(json.loads(ballet))
    parsed_soc_data["full_voting_profile"] = full_voting_profile
    parsed_soc_data_list.append(parsed_soc_data)


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
            approval_vector = [0]*num_candidates
            print("actual_voter_ballet ", actual_voter_ballet, actual_voter_ballet[:approval_count])
            for cand in actual_voter_ballet[:approval_count]:
                actual_winning_score_dict[cand] += 1
        print(actual_winning_score_dict, "approval count ", approval_count)
        # actual_winner is the candidate who was chosen by most number voters with tie breaking (randomly chosen)
        actual_highest_vote_per_approval[approval_count] += max(actual_winning_score_dict.values())

    actual_highest_vote_per_approval[approval_count] /= avg_runs

        # actual_winning_candidate_list = list(filter(lambda x: actual_winning_score_dict[x] == actual_highest_vote_per_approval[approval_count], actual_winning_score_dict))
        # # print("actual_winning_candidate_list: ", actual_winning_candidate_list)
        # actual_winning_candidate = random.choice(actual_winning_candidate_list)
print("actual_highest_vote_per_approval ", actual_highest_vote_per_approval)


output_file = "results_soc_data.json"
result = {}

for approval_count in approval_count_list:
    result[approval_count] = {}
    result[approval_count]["test_0"] = {}
    result[approval_count]["test_0"]["avg_score_arr"] = [actual_highest_vote_per_approval[approval_count]]*int(iterations / batch)

num_iter_arr = [i for i in range(batch, iterations + 1, batch)]

for approval_count in approval_count_list:
    print("approval count - ", approval_count)
    for key in input_conf.keys():
        print("key - ", key)
        result[approval_count][key] = {}
        avg_score_arr = []
        score_arr = []
        for i in tqdm(range(avg_runs)):
            print("avg run - ", i)
            train = train_model(iterations, batch, parsed_soc_data_list[i], approval_count, input_conf[key]["epsilon"], input_conf[key]["grad_epsilon"], input_conf[key]["epsilon_final"], input_conf[key]["epsilon_decay"])   # for each iteration returns a dictionary containing voter and there fnal reward i.e borda score of top candidate
            score, welfare_dict_list = train.train(1, input_conf[key]["single_iterative_voting"])
            score_arr.append(score)
        score_arr = np.array(score_arr)
        avg_score_arr = score_arr.sum(axis=0)
        avg_score_arr = avg_score_arr/avg_runs
        result[approval_count][key]["avg_score_arr"] = avg_score_arr.tolist()
        result[approval_count][key]["welfare_dict_list"] = welfare_dict_list

    plot = plt.figure()
    for key in result[approval_count].keys():
        print(result[approval_count][key]["avg_score_arr"][0], result[approval_count][key]["avg_score_arr"][-1])
        plt.plot(num_iter_arr, result[approval_count][key]["avg_score_arr"], label=key)

    plt.legend(loc='upper right')
    plt.xlabel("Number of iterations")
    plt.ylabel("Avg Approval Score")
    plt.ylim(0, actual_highest_vote_per_approval[approval_count] + 3)
    plt.show()
    plt.savefig('results/approval_count_' + str(approval_count) + '_' + data_file_name["name"] + \
                 "_" + str(iterations) + "_" + data_file_name["test_case"] + '.png')

with open(output_file, "w") as f:
    json.dump(result, f)



# print(num_iter_arr, len(avg_score_arr))
# print(np.shape(avg_score_arr))
# num_iter_arr.pop()

# print("avg_score_arr: ", avg_score_arr)
# print("num_iter_arr: ", num_iter_arr)



# borda_score_per_cand = {}  # dictionary conating list of borda score over different iteration per cand
# for cand in range(num_candidates):
#     borda_score_per_cand[cand] = []


# for score in borda_scores_arr:
#     for cand in range(num_candidates):
#         borda_score_per_cand[cand].append(score[cand])

# print(borda_score_per_cand)

# plot = plt.figure()
# for key in result.keys():
#     plt.plot(num_iter_arr, result[key]["avg_score_arr"], label=key)
# # plt.plot(num_iter_arr, [actual_winning_score_dict[0]]*len(num_iter_arr), "r--")
# # plt.plot(num_iter_arr, borda_score_per_cand[1], "g", label="candidate 1")
# # plt.plot(num_iter_arr, [actual_winning_score_dict[1]]*len(num_iter_arr), "g--")
# # plt.plot(num_iter_arr, borda_score_per_cand[2], "b", label="candidate 2")
# # plt.plot(num_iter_arr, [actual_winning_score_dict[2]]*len(num_iter_arr), "b--")
# plt.legend()
# plt.xlabel("Number of iterations")
# plt.ylabel("Avg Borda score of learning agents (Plurality)")
# plt.ylim(0, actual_highest_vote+5)
# plt.show()
# plt.savefig('res.png')