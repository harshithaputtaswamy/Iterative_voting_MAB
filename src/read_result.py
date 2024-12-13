import os
import ujson
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

# num_iter_arr = [i for i in range(batch, iterations + 1, batch)]

# plot = plt.figure()
# for key in result["graph_coords"].keys():
#     print(result["graph_coords"][key]["avg_score_arr"][0], result["graph_coords"][key]["avg_score_arr"][-1])
#     plt.plot(num_iter_arr, result["graph_coords"][key]["avg_score_arr"], label=key)
#     # plt.text("epsilon - {}, epsilon decay - {}".format(input_conf[key]["epsilon"], input_conf[key]["epsilon_decay"]))

# plt.legend(loc='upper right')
# plt.xlabel("Number of iterations")
# plt.ylabel("Avg Borda Score with {}".format(voting_rule))
# # plt.ylim(0, actual_highest_vote_per_approval + 1)
# plt.show()
# plt.savefig('results/setting_{}_{}_voter_'.format(voting_setting, voting_rule) + str(num_voters) + '_cand_' + str(num_candidates) + \
#                 "_committee_size_" + str(committee_size) + "_iter_" + str(iterations) + "_avg_" + str(avg_runs) + '.png')

class read_results():
    def __init__(self, file_to_read):
        self.file_to_read = file_to_read

        
        with open(file_to_read, "r") as result_file:
            result = ujson.load(result_file)
        self.result = result

        self.run_setting = self.result["run_setting"]
        self.test_details = self.run_setting["test_details"]

    def get_run_setting(self):
        return self.run_setting
    
    def get_avg_score_graph_coords(self):
        return self.result["graph_coords"]


    def get_preferences(self):
        preferences = {}
        for test in self.test_details.keys():
            curr_test = self.test_details[test]
            preferences[test] = {}
            for run in curr_test.keys():
                curr_run = curr_test[run]
                preferences[test][run] = curr_run["predicted_prefrence_values"]["profile"]     # preference = [ith iter {voter: [preference]}], list of voter preference dictionary over i iterations
        return preferences


    def get_rewards(self):
        rewards = {}
        for test in self.test_details.keys():
            curr_test = self.test_details[test]
            rewards[test] = {}
            for run in curr_test.keys():
                curr_run = curr_test[run]
                rewards[test][run] = curr_run["predicted_prefrence_values"]["rewards"]     # preference = [ith iter {voter: [preference]}], list of voter preference dictionary over i iterations
        return rewards


    def get_tie_preferences(self):
        preferences = {}
        for test in self.test_details.keys():
            curr_test = self.test_details[test]
            preferences[test] = {}
            print(curr_test.keys())
            for tie_breaking_rule in curr_test.keys():
                preferences[test][tie_breaking_rule] = {}
                for run in curr_test[tie_breaking_rule].keys():
                    curr_run = curr_test[tie_breaking_rule][run]
                    preferences[test][tie_breaking_rule][run] = curr_run["predicted_prefrence_values"]["profile"]     # preference = [ith iter {voter: [preference]}], list of voter preference dictionary over i iterations
        return preferences


    def get_voting_rules_comp_preferences(self):
        preferences = {}
        for voting_rule in self.test_details.keys():
            curr_voting_rule = self.test_details[voting_rule]
            preferences[voting_rule] = {}
            for run in curr_voting_rule.keys():
                curr_run = curr_voting_rule[run]
                preferences[voting_rule][run] = curr_run["predicted_prefrence_values"]["profile"]     # preference = [ith iter {voter: [preference]}], list of voter preference dictionary over i iterations
        return preferences
    

    def get_voting_rules_comp_rewards(self):
        rewards = {}
        for voting_rule in self.test_details.keys():
            curr_test = self.test_details[voting_rule]
            rewards[voting_rule] = {}
            for run in curr_test.keys():
                curr_run = curr_test[run]
                rewards[voting_rule][run] = curr_run["predicted_prefrence_values"]["rewards"]     # preference = [ith iter {voter: [preference]}], list of voter preference dictionary over i iterations
        return rewards