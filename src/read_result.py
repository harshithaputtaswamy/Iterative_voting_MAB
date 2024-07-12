import os
import json
import matplotlib.pyplot as plt


# voting_setting = 1  # 0 - single winner setting, 1 - committee voting
# committee_size_list = [2, 3]  # ( k will be the committee size)
# committee_size = 3
# num_voters = 10
# num_candidates = 5


# # voting_rule = "plurality"
# # voting_rule = "borda"
# voting_rule = "borda_top_cand"
# # voting_rule = "approval"
# # voting_rule = "copeland"


# avg_runs = 5
# iterations = 50000
# batch = 1000


curr_dir = os.path.dirname(os.getcwd())
output_file = "setting_1_borda_top_cand_voter_10_cand_5_committee_size_2_iter_20000_avg_10.json"

output_path = os.path.join(curr_dir + "/numerical_results/" + "borda_top_cand" + "/", output_file)


result_file = open(output_path)
result = json.load(result_file)


run_setting = result["run_setting"]
voting_setting = run_setting["voting_setting"]  # 0 - single winner setting, 1 - committee voting
committee_size = run_setting["committee_size"]
num_voters = run_setting["num_voters"]
num_candidates = run_setting["num_candidates"]
voting_rule = run_setting["voting_rule"]
avg_runs = run_setting["avg_runs"]
iterations = run_setting["iterations"]
batch = run_setting["batch"]


test_details = run_setting["test_details"]

test_config = result["config"]

num_iter_arr = [i for i in range(batch, iterations + 1, batch)]

plot = plt.figure()
for key in result["graph_coords"].keys():
    print(result["graph_coords"][key]["avg_score_arr"][0], result["graph_coords"][key]["avg_score_arr"][-1])
    plt.plot(num_iter_arr, result["graph_coords"][key]["avg_score_arr"], label=key)
    # plt.text("epsilon - {}, epsilon decay - {}".format(input_conf[key]["epsilon"], input_conf[key]["epsilon_decay"]))

plt.legend(loc='upper right')
plt.xlabel("Number of iterations")
plt.ylabel("Avg Borda Score with {}".format(voting_rule))
# plt.ylim(0, actual_highest_vote_per_approval + 1)
plt.show()
plt.savefig('results/setting_{}_{}_voter_'.format(voting_setting, voting_rule) + str(num_voters) + '_cand_' + str(num_candidates) + \
                "_committee_size_" + str(committee_size) + "_iter_" + str(iterations) + "_avg_" + str(avg_runs) + '.png')


def get_preferences():
    preferences = {}
    for test in test_details.keys():
        curr_test = test_details[test]
        preferences[test] = {}
        for run in curr_test.keys():
            curr_run = curr_test[run]
            preferences[test][run] = curr_run["predicted_prefrence_values"]["profile"]     # preference = [ith iter {voter: [preference]}], list of voter preference dictionary over i iterations
    return preferences, iterations