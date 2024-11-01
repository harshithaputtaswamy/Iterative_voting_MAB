import os
import json
from scipy import stats
import matplotlib.pyplot as plt
from read_result import get_preferences



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




file_to_read = "setting_1_approval_tie_breaking_rand_approval_count_2_voter_10_cand_5_committee_size_3_iter_50000_avg_50.json"
preferences, run_setting = get_preferences(file_to_read, "approval")
num_voters = run_setting["num_voters"]
voting_setting = run_setting["voting_setting"]
committee_size = run_setting["committee_size"]
num_candidates = run_setting["num_candidates"]
voting_rule = run_setting["voting_rule"]
avg_runs = run_setting["avg_runs"]
iterations = run_setting["iterations"]
approval_count = run_setting["approval_count"]
tie_breaking_rule = run_setting["tie_breaking_rule"]
kt_dict_interval = {}

window_size = 10
end_interval = iterations - iterations


def kt_distance(preference_dict_t, preference_dict_t_n, num_voters):
	total_kt = 0
	for voter in range(num_voters):
		res = stats.kendalltau(preference_dict_t[str(voter)], preference_dict_t_n[str(voter)])
		total_kt += res.statistic       # res.statistic closer to 1 then it implies strong positive ordinal correlation, 0 implies weak ordinal correlation, -1 implies strong negative ordinal correlation
		
	return total_kt/num_voters


def sliding_kt_distance(t):
	kt_dict = {}
	for test in preferences.keys():
		# print(test)
		kt_dict[test] = {}
		avg_run_kt = [0]*window_size
		kt = 0
		for tie_breaking_rule in preferences[test].keys():
			kt_dict[test][tie_breaking_rule] = []
			for run in preferences[test][tie_breaking_rule].keys():
				curr_pref = preferences[test][tie_breaking_rule][run][abs(t - window_size) : t]
				kt_iters = []
				for pref in curr_pref:
					kt_iters.append(kt_distance(preferences[test][tie_breaking_rule][run][t-1], pref, num_voters))
				for i in range(window_size):
					avg_run_kt[i] += kt_iters[i]        # sum over different runs

			for i in range(window_size):
				avg_run_kt[i] /= avg_runs
			kt_dict[test][tie_breaking_rule] = sum(avg_run_kt) / window_size       # get sum over the window
			# kt_dict[test][tie_breaking_rule] = max(avg_run_kt) / window_size     # get max over the window
		return kt_dict


for i in range(iterations, end_interval, -window_size):
	print(i)
	kt_dict = sliding_kt_distance(i)
	if not kt_dict_interval:
		for test in kt_dict.keys():
			for tie_breaking_rule in kt_dict[test].keys():
				kt_dict_interval[test][tie_breaking_rule] = [kt_dict[test][tie_breaking_rule]]
	else:
		for test in kt_dict.keys():
			for tie_breaking_rule in kt_dict[test].keys():
				kt_dict_interval[test][tie_breaking_rule].append(kt_dict[test][tie_breaking_rule])
	

plot = plt.figure()

for test in kt_dict_interval.keys():
	for tie_breaking_rule in kt_dict_interval[test].keys():
		plt.plot(range(iterations - 1, end_interval, -window_size), kt_dict_interval[test][tie_breaking_rule], label=tie_breaking_rule)

	curr_dir = os.path.dirname(os.getcwd())
	os.makedirs(curr_dir + "/graph_results/" + voting_rule, exist_ok=True)

	plt.legend(loc='upper right')
	plt.xlabel("Number of iterations")
	plt.ylabel("Avg kendalltau corelation - ")
	plt.show()

	if voting_rule == "approval":
		graph_file = 'kt_distance_setting_{}_{}_test_config_{}_approval_count_{}_voter_'.format(voting_setting, voting_rule, test, approval_count) + str(num_voters) + '_cand_' + str(num_candidates) + \
					"_committee_size_" + str(committee_size) + "_iter_" + str(iterations) + "_to_" + str(end_interval) + "_avg_" + str(avg_runs) + "_window_size_" + str(window_size) + '.png'
	else:
		graph_file = 'kt_distance_setting_{}_{}_test_config_{}_voter_'.format(voting_setting, voting_rule, test) + str(num_voters) + '_cand_' + str(num_candidates) + \
					"_committee_size_" + str(committee_size) + "_iter_" + str(iterations) + "_to_" + str(end_interval) + "_avg_" + str(avg_runs) + "_window_size_" + str(window_size) + '.png'
	graph_path = os.path.join(curr_dir + "/graph_results/" + voting_rule + "/", graph_file)

	plt.savefig(graph_path)



# get kt dist over 0-50000 iters, tie breaking using disctioanry order
# July 29 - 14, next meet 16