import os
import json
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
from read_result import read_results


def get_borda_score(curr_voting_profile, num_alternatives):
	# Computes the borda score for the votings casted until now

	res = dict([])
	for order, multiplicity in curr_voting_profile.items():
		i = num_alternatives
		order_arr = json.loads(order)
		for alt in order_arr:
			i -= 1
			if alt not in res:
				res[alt] = i * multiplicity
			else:
				res[alt] += i * multiplicity
			
	return res


curr_dir = os.path.dirname(os.getcwd())
voting_rules_list_dict = {
    "score_based_rules_comp_study" : ["borda", "borda_top_cand", "approval", "plurality", "anti_plurality"],
    "condorcet_rules_comp_study" : ["chamberlin_courant", "bloc", "monroe", "stv", "pav"]
}

for voting_rule_type in voting_rules_list_dict.keys():

	file_to_read = "setting_1_test_config_test_3_tie_breaking_rule_rand_approval_count_2_voter_10_cand_5_committee_size_3_iter_50000_avg_50.json"
	file_to_read = curr_dir + "/numerical_results/{}/".format(voting_rule_type) + file_to_read


	read_result = read_results(file_to_read)
	run_setting = read_result.get_run_setting()
	preferences = read_result.get_voting_rules_comp_preferences()
	rewards = read_result.get_voting_rules_comp_rewards()

	config = run_setting["config"]
	num_voters = run_setting["num_voters"]
	voting_setting = run_setting["voting_setting"]
	committee_size = run_setting["committee_size"]
	num_candidates = run_setting["num_candidates"]
	voting_rule_list = run_setting["voting_rule_list"]
	avg_runs = run_setting["avg_runs"]
	iterations = run_setting["iterations"]
	approval_count = run_setting["approval_count"]
	tie_breaking_rule = run_setting["tie_breaking_rule"]
	test = list(config.keys())[0]

	kt_dict_interval = {}
	# for voting_rule in voting_rule_list:
	# 	kt_dict_interval[voting_rule] = []

	window_size = 10
	end_interval = iterations - iterations


	def kt_distance(voting_rule, preference_dict_t, preference_dict_t_n, num_voters):
		total_kt = 0
		for voter in range(num_voters):
			if voting_rule == "plurality" or voting_rule == 'anti_plurality':
				total_kt += 0 if preference_dict_t[str(voter)] != preference_dict_t_n[str(voter)] else 1
			else:
				res = stats.kendalltau(preference_dict_t[str(voter)], preference_dict_t_n[str(voter)])
				total_kt += res.statistic       # res.statistic closer to 1 then it implies strong positive ordinal correlation, 0 implies weak ordinal correlation, -1 implies strong negative ordinal correlation
			
		return total_kt/num_voters


	def sliding_kt_distance(t):
		kt_dict = {}
		for voting_rule in voting_rule_list:
			avg_run_kt = [0]*window_size
			kt_dict[voting_rule] = []
			for run in preferences[voting_rule].keys():
				curr_pref = preferences[voting_rule][run][abs(t - window_size) : t]
				kt_iters = []
				for pref in curr_pref:
					kt_iters.append(kt_distance(voting_rule, preferences[voting_rule][run][t-1], pref, num_voters))
				for i in range(window_size):
					avg_run_kt[i] += kt_iters[i]        # sum over different runs

			for i in range(window_size):
				avg_run_kt[i] /= avg_runs
			kt_dict[voting_rule] = sum(avg_run_kt) / window_size       # get sum over the window
		return kt_dict
	

	for i in range(iterations, end_interval, -window_size):
		# print(i)

		kt_dict = sliding_kt_distance(i)
		# for test in kt_dict.keys():
		for voting_rule in kt_dict.keys():
			if voting_rule not in kt_dict_interval:
				kt_dict_interval[voting_rule] = [kt_dict[voting_rule]]
			else:
				kt_dict_interval[voting_rule].append(kt_dict[voting_rule])
	plot = plt.figure()

	# for test in kt_dict_interval.keys():
	for voting_rule in kt_dict_interval.keys():
		data_series = pd.Series(kt_dict_interval[voting_rule])
		smoothed_data = data_series.rolling(window=100, min_periods=1).mean()
		plt.plot(range(iterations - 1, end_interval, -window_size), smoothed_data, label=voting_rule)

	curr_dir = os.path.dirname(os.getcwd())
	os.makedirs(curr_dir + "/graph_results/{}/".format(voting_rule_type), exist_ok=True)

	plt.legend(loc='upper right')
	plt.xlabel("Number of iterations")
	plt.ylabel("Avg kendalltau corelation - ")
	plt.show()

	graph_file = 'kt_distance_setting_{}_test_config_{}_tie_breaking_rule_{}_voter_'.format(voting_setting, test, tie_breaking_rule) + str(num_voters) + '_cand_' + str(num_candidates) + \
		"_committee_size_" + str(committee_size) + "_iter_" + str(iterations) + "_to_" + str(end_interval) + "_avg_" + str(avg_runs) + "_window_size_" + str(window_size) + '.png'
	graph_path = os.path.join(curr_dir + "/graph_results/{}/".format(voting_rule_type), graph_file)

	plt.savefig(graph_path)



	def cost_of_strategy():
		cost_dict = {}
		for voting_rule in rewards.keys():
			cost_dict[voting_rule] = []
			avg_run_cost = [0]*iterations
			for run in rewards[voting_rule].keys():
				true_reward = rewards[voting_rule][run][0]
				for iter in range(iterations):
					#get the difference between 0th borda score and nth borda score
					avg_run_cost[iter] += (true_reward - rewards[voting_rule][run][iter])
			for iter in range(iterations):
				avg_run_cost[iter] /= avg_runs
			cost_dict[voting_rule] = avg_run_cost
		return cost_dict


	# plot cost of strgategy
	cost_of_strategy_dict = cost_of_strategy()

	plot = plt.figure()

	for test in cost_of_strategy_dict.keys():
		data_series = pd.Series(cost_of_strategy_dict[test])
		smoothed_data = data_series.rolling(window=100, min_periods=1).mean()
		plt.plot(range(len(smoothed_data)), smoothed_data, label=test)

	curr_dir = os.path.dirname(os.getcwd())
	os.makedirs(curr_dir + "/graph_results/{}/".format(voting_rule_type), exist_ok=True)

	plt.legend(loc='upper right')
	plt.xlabel("Number of iterations")
	plt.ylabel("Avg cost of strategy")
	plt.show()
	graph_file = 'cost_of_strategy_setting_{}_tie_breaking_{}_approval_count_{}_voter_'.format(voting_setting, tie_breaking_rule, approval_count) + str(num_voters) + '_cand_' + str(num_candidates) + \
				"_committee_size_" + str(committee_size) + "_iter_" + str(iterations) + "_avg_" + str(avg_runs) + '.png'
	graph_path = os.path.join(curr_dir + "/graph_results/{}/".format(voting_rule_type), graph_file)

	plt.savefig(graph_path)





	# get kt dist over 0-50000 iters, tie breaking using disctioanry order
	# July 29 - 14, next meet 16