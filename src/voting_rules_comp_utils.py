import os
import json
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
from read_result import read_results
from matplotlib.ticker import FormatStrFormatter


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


voting_rules_labels_utils = {
    "borda" : "Borda", 
    "borda_top_cand" : "Borda Top Cand", 
    "chamberlin_courant" :  r"$\beta-CC$", 
    "monroe" : "Monroe", 
    "stv" : "STV", 
    "pav" : "PAV", 
    "bloc" : "Bloc",
    "approval" : "Approval",
    "plurality" : "Plurality",
    "anti_plurality" : "Anti-Plurality",
}

voting_rules_labels = {
    "borda" : r"$Borda_S$", 
    "borda_top_cand" : r"$Borda Top Cand_S$", 
    "chamberlin_courant" : r"$\beta-CC_S$", 
    "monroe" : r"$Monroe_S$", 
    "stv" : r"$STV_S$", 
    "pav" : r"$PAV_S$", 
    "bloc" : r"$Bloc_S$",
    "approval" : r"$Approval_S$",
    "plurality" : r"$Plurality_S$",
    "anti_plurality" : r"$Anti-Plurality_S$",
    "true_borda" : r"$Borda_T$", 
    "true_borda_top_cand" : r"$Borda Top Cand_T$", 
    "true_chamberlin_courant" : r"$\beta-CC_T$", 
    "true_monroe" : r"$Monroe_T$", 
    "true_stv" : r"$STV_T$", 
    "true_pav" : r"$PAV_T$", 
    "true_bloc" : r"$Bloc_T$",
    "true_approval" : r"$Approval_T$",
    "true_plurality" : r"$Plurality_T$",
    "true_anti_plurality" : r"$Anti-Plurality_T$"
}

colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2']


voting_rules_list_dict = {
    "plurality_ballot_rules" : ["plurality", "anti_plurality"],
    "approval_ballot_rules" : ["approval", "pav", "bloc"],
    "ranked_ballot_rules" : ["borda", "chamberlin_courant", "monroe", "stv"]
}






file_to_read = "setting_1_test_config_test_3_tie_breaking_rule_rand_approval_count_2_voter_10_cand_5_committee_size_3_iter_50000_avg_50.json"

for voting_rule_type in voting_rules_list_dict.keys():

	file_to_read = curr_dir + "/numerical_results/voting_rules_comp_study/{}/".format(voting_rule_type) + file_to_read

	read_result = read_results(file_to_read)
	run_setting = read_result.get_run_setting()
	preferences = read_result.get_voting_rules_comp_preferences()
	rewards = read_result.get_voting_rules_comp_rewards()
	avg_score_coords = read_result.get_avg_score_graph_coords()

	config = run_setting["config"]
	num_voters = run_setting["num_voters"]
	voting_setting = run_setting["voting_setting"]
	committee_size = run_setting["committee_size"]
	num_candidates = run_setting["num_candidates"]
	voting_rule_list = run_setting["voting_rule_list"]
	avg_runs = run_setting["avg_runs"]
	iterations = run_setting["iterations"]
	batch = run_setting["batch"]
	approval_count = run_setting["approval_count"]
	tie_breaking_rule = run_setting["tie_breaking_rule"]
	test = list(config.keys())[0]

	kt_dict_interval = {}
	# for voting_rule in voting_rule_list:
	# 	kt_dict_interval[voting_rule] = []

	window_size = 10
	end_interval = 0
	print(iterations)


	def kt_distance(voting_rule, preference_dict_t, preference_dict_t_n, num_voters):
		total_kt = 0
		for voter in range(num_voters):
			if voting_rule in ["plurality", "anti_plurality"]:
				total_kt += 0 if preference_dict_t[str(voter)] != preference_dict_t_n[str(voter)] else 1
			else:
				res = stats.kendalltau(preference_dict_t[str(voter)], preference_dict_t_n[str(voter)])
				total_kt += res.statistic       # res.statistic closer to 1 then it implies strong positive ordinal correlation, 0 implies weak ordinal correlation, -1 implies strong negative ordinal correlation
			
		return total_kt/num_voters


	def sliding_kt_distance(t):
		kt_dict = {}
		for voting_rule in voting_rule_list:
			if voting_rule != "borda_top_cand":
				avg_run_kt = [0]*(window_size - 1)
				kt_dict[voting_rule] = []
				for run in preferences[voting_rule].keys():
					curr_pref = preferences[voting_rule][run][t - window_size + 1: t]
					# kt_iters = [0]*(window_size)
					# for pref in curr_pref:
					# 	kt_iters.append(kt_distance(voting_rule, preferences[voting_rule][run][t], pref, num_voters))
					for i in range(window_size - 1):
						avg_run_kt[i] += kt_distance(voting_rule, preferences[voting_rule][run][t], curr_pref[i], num_voters)        # sum over different runs
						# print(avg_run_kt[i])

				for i in range(window_size - 1):
					avg_run_kt[i] /= avg_runs
				kt_dict[voting_rule] = sum(avg_run_kt) / (window_size - 1)       # get sum over the window
		return kt_dict
	

	for i in range(iterations - 1, end_interval, -window_size):
		print(i)

		kt_dict = sliding_kt_distance(i)
		# for test in kt_dict.keys():
		for voting_rule in kt_dict.keys():
			if voting_rule not in kt_dict_interval:
				kt_dict_interval[voting_rule] = [kt_dict[voting_rule]]
			else:
				kt_dict_interval[voting_rule].append(kt_dict[voting_rule])
	
	fig, ax = plt.subplots()

	# for test in kt_dict_interval.keys():
	for voting_rule in kt_dict_interval.keys():
		data_series = pd.Series(kt_dict_interval[voting_rule])
		smoothed_data = data_series.rolling(window=50, center=True, min_periods=1).mean()
		ax.plot(range(iterations - 1, end_interval, -window_size), smoothed_data.to_list(), label=voting_rules_labels_utils[voting_rule])

	curr_dir = os.path.dirname(os.getcwd())
	os.makedirs(curr_dir + "/graph_results/voting_rules_comp_study/{}/".format(voting_rule_type), exist_ok=True)

	plt.text(.9, .01, "n: {}, m: {}, k: {}, approval count: {}".format(num_voters, num_candidates, committee_size, approval_count),
		fontsize=10, ha='right', va='bottom', transform=ax.transAxes)

	plt.legend(loc='upper right')
	plt.xlabel("Number of iterations", fontsize=12)
	plt.ylabel("Avg. Kendall-Tau correlation", fontsize=12)
	plt.gca().yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
	# plt.tight_layout()
	plt.show()

	graph_file = 'kt_distance_setting_{}_test_config_{}_tie_breaking_rule_{}_voter_'.format(voting_setting, test, tie_breaking_rule) + str(num_voters) + '_cand_' + str(num_candidates) + \
		"_committee_size_" + str(committee_size) + "_iter_" + str(iterations) + "_to_" + str(end_interval) + "_avg_" + str(avg_runs) + "_window_size_" + str(window_size) + '.png'
	graph_path = os.path.join(curr_dir + "/graph_results/voting_rules_comp_study/{}/".format(voting_rule_type), graph_file)

	plt.savefig(graph_path)



	def cost_of_strategy():
		cost_dict = {}
		for voting_rule in voting_rule_list:
			if voting_rule != "borda_top_cand":
				print("voting_rule: ", voting_rule)
				cost_dict[voting_rule] = []
				avg_run_cost = [0]*iterations
				for run in rewards[voting_rule].keys():
					true_reward = rewards[voting_rule][run][0]
					for iter in range(iterations):
						#get the difference between 0th borda score and nth borda score
						# avg_run_cost[iter] += (true_reward - rewards[voting_rule][run][iter])

						# get the ratio of borda score with stratigic behaviour to true preference
						avg_run_cost[iter] += rewards[voting_rule][run][iter] / true_reward
				for iter in range(iterations):
					avg_run_cost[iter] /= avg_runs
				cost_dict[voting_rule] = avg_run_cost
		return cost_dict


	# plot cost of strgategy
	cost_of_strategy_dict = cost_of_strategy()

	fig, ax = plt.subplots()

	for voting_rule in cost_of_strategy_dict.keys():
		data_series = pd.Series(cost_of_strategy_dict[voting_rule])
		smoothed_data = data_series.rolling(window=100, min_periods=1).mean()
		ax.plot(range(len(smoothed_data)), smoothed_data, label=voting_rules_labels_utils[voting_rule])

	curr_dir = os.path.dirname(os.getcwd())
	os.makedirs(curr_dir + "/graph_results/{}/".format(voting_rule_type), exist_ok=True)
	
	plt.text(.9, .01, "n: {}, m: {}, k: {}, approval count: {}".format(num_voters, num_candidates, committee_size, approval_count),
		fontsize=10, ha='right', va='bottom', transform=ax.transAxes)

	plt.legend(loc='upper right')
	plt.xlabel("Number of iterations", fontsize=12)
	plt.ylabel("Avg. cost of strategy " + r"$\frac{\beta(r_S)}{\beta(r_T)}$", fontsize=12)
	plt.gca().yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
	plt.tight_layout()
	plt.show()
	graph_file = 'cost_of_strategy_setting_{}_tie_breaking_{}_approval_count_{}_voter_'.format(voting_setting, tie_breaking_rule, approval_count) + str(num_voters) + '_cand_' + str(num_candidates) + \
				"_committee_size_" + str(committee_size) + "_iter_" + str(iterations) + "_avg_" + str(avg_runs) + '.png'
	graph_path = os.path.join(curr_dir + "/graph_results/voting_rules_comp_study/{}/".format(voting_rule_type), graph_file)

	plt.savefig(graph_path)



	# # graphs for showing avg borda score for given voting setting 
	# fig, ax = plt.subplots()
	# idx = 0
	# for voting_rule in voting_rule_list:
	# 	if voting_rule != "borda_top_cand":
	# 		print(voting_rule)
	# 		print(avg_score_coords[voting_rule]["avg_score_arr"]["true"][0], avg_score_coords[voting_rule]["avg_score_arr"]["strategic"][-1])
	# 		data_series = pd.Series(avg_score_coords[voting_rule]["avg_score_arr"]["strategic"])
	# 		smoothed_data = data_series.rolling(window=100, min_periods=1).mean()
	# 		plt.plot([i for i in range(batch, iterations + 1, batch)], smoothed_data, label=voting_rules_labels[voting_rule], color=colors[idx], linestyle='-')
	# 		plt.plot([i for i in range(batch, iterations + 1, batch)], avg_score_coords[voting_rule]["avg_score_arr"]["true"], label=voting_rules_labels["true_{}".format(voting_rule)], color=colors[idx], linestyle='--')
	# 		idx += 1

	# plt.text(.9, .01, "n: {}, m: {}, k: {}, approval count: {}".format(num_voters, num_candidates, committee_size, approval_count),
	# 	fontsize=10, ha='right', va='bottom', transform=ax.transAxes)

	# plt.legend(loc='upper right')
	# plt.xlabel("Number of iterations", fontsize=12)
	# plt.ylabel("Avg. Borda Score", fontsize=12)
	# plt.gca().yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
	# # plt.tight_layout()
	# plt.show()
	# graph_file = 'avg_score_setting_{}_test_config_{}_tie_breaking_rule_{}_approval_count_{}_voter_'.format(voting_setting, test, tie_breaking_rule, approval_count) + str(num_voters) + '_cand_' + str(num_candidates) + \
	# 			"_committee_size_" + str(committee_size) + "_iter_" + str(iterations) + "_avg_" + str(avg_runs) + '.png'
	# graph_path = os.path.join(curr_dir + "/graph_results/voting_rules_comp_study/{}/".format(voting_rule_type), graph_file)

	# plt.savefig(graph_path)


