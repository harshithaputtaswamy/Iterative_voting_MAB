from train import *
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
borda_scores_arr = []
num_iter_arr = []


# import matplotlib
# matplotlib.use("QtAgg")

# data_file = open("parsed_soc_data.json")
# parsed_soc_data = json.load(data_file)

data_file = open("sanity_test_data.json")
parsed_soc_data = json.load(data_file)["test2"]

num_voters = parsed_soc_data["num_voters"]
num_candidates = parsed_soc_data["num_candidates"]
flattened_voting_profile = parsed_soc_data["flattened_voting_profile"]

full_voting_profile = []
for ballet, num in flattened_voting_profile.items():
    for i in range(num):
        full_voting_profile.append(json.loads(ballet))
parsed_soc_data["full_voting_profile"] = full_voting_profile

# calculate borda score for this all candidate to see who the actual winner would be
actual_winning_borda_score_dict = {}
for cand in range(num_candidates):
    winning_borda_score = 0
    for voter in range(num_voters):
        actual_voter_ballet = full_voting_profile[voter]
        borda_score = num_candidates - 1 - actual_voter_ballet.index(cand)
        winning_borda_score += borda_score
    actual_winning_borda_score_dict[cand] = winning_borda_score

print("actual winning candidate borda score: {}", actual_winning_borda_score_dict)

# actual_winner is the candidate who was chosen by most number voters with tie breaking (randomly chosen)
actual_highest_vote = max(actual_winning_borda_score_dict.values())
actual_winning_candidate_list = list(filter(lambda x: actual_winning_borda_score_dict[x] == actual_highest_vote, actual_winning_borda_score_dict))
# print("actual_winning_candidate_list: ", actual_winning_candidate_list)
actual_winning_candidate = random.choice(actual_winning_candidate_list)
# print("winner ", actual_winning_candidate)

# for i in range(50, 500, 50):
#     avg_borda_score = 0
#     # for j in range(i):
#     #     avg_borda_score += train(1000, 1, 0.5, parsed_soc_data)   # for each iteration returns a dictionary containing voter and there fnal reward i.e borda score of top candidate

input_file = open("config.json")
input_conf = json.load(input_file)

result = {}
output_file = open("results.json")
result = json.load(output_file)

avg_runs = 10
iterations = 20000
batch = 50

for key in input_conf.keys():
    result[key] = {}
    avg_borda_score_arr = []
    borda_scores_arr = []
    for i in tqdm(range(avg_runs)):
        train = train_model(iterations, batch, parsed_soc_data, input_conf[key]["epsilon"], input_conf[key]["grad_epsilon"], input_conf[key]["epsilon_final"], input_conf[key]["epsilon_decay"])   # for each iteration returns a dictionary containing voter and there fnal reward i.e borda score of top candidate
        borda_scores_arr.append(train.train(1, input_conf[key]["single_iterative_voting"]))
        # print(i)
    borda_scores_arr = np.array(borda_scores_arr)
    avg_borda_score_arr = borda_scores_arr.sum(axis=0)
    avg_borda_score_arr = avg_borda_score_arr/avg_runs
    result[key]["avg_borda_score_arr"] = avg_borda_score_arr.tolist()

with open("results.json", "w") as f:
    json.dump(result, f)

num_iter_arr = [i for i in range(50, iterations + 1, batch)]

# print("avg_borda_score_arr: ", avg_borda_score_arr)
# print("num_iter_arr: ", num_iter_arr)



# borda_score_per_cand = {}  # dictionary conating list of borda score over different iteration per cand
# for cand in range(num_candidates):
#     borda_score_per_cand[cand] = []


# for score in borda_scores_arr:
#     for cand in range(num_candidates):
#         borda_score_per_cand[cand].append(score[cand])

# print(borda_score_per_cand)

plot = plt.figure()
for key in result.keys():
    plt.plot(num_iter_arr, result[key]["avg_borda_score_arr"], label=key)
# plt.plot(num_iter_arr, [actual_winning_borda_score_dict[0]]*len(num_iter_arr), "r--")
# plt.plot(num_iter_arr, borda_score_per_cand[1], "g", label="candidate 1")
# plt.plot(num_iter_arr, [actual_winning_borda_score_dict[1]]*len(num_iter_arr), "g--")
# plt.plot(num_iter_arr, borda_score_per_cand[2], "b", label="candidate 2")
# plt.plot(num_iter_arr, [actual_winning_borda_score_dict[2]]*len(num_iter_arr), "b--")
plt.legend()
plt.xlabel("Number of iterations")
plt.ylabel("Avg Borda score of learning agents (Plurality)")
plt.ylim(0, 40)
plt.show()
plt.savefig('res.png')