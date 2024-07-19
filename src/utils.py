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




file_to_read = "setting_1_borda_voter_10_cand_5_committee_size_3_iter_50000_avg_50.json"
preferences, run_setting = get_preferences(file_to_read, "borda")
num_voters = run_setting["num_voters"]
voting_setting = run_setting["voting_setting"]
committee_size = run_setting["committee_size"]
num_candidates = run_setting["num_candidates"]
voting_rule = run_setting["voting_rule"]
avg_runs = run_setting["avg_runs"]
iterations = run_setting["iterations"]
kt_dict_interval = {}

window_size = 100
end_interval = iterations - iterations


def kt_distance(preference_dict_t, preference_dict_t_n, num_voters):
	total_kt = 0
	for voter in range(num_voters):
		res = stats.kendalltau(preference_dict_t[str(voter)], preference_dict_t_n[str(voter)])
		total_kt += res.statistic       # res.statistic closer to 1 then it implies strong positive ordinal correlation, 0 implies weak ordinal correlation, -1 implies strong negative ordinal correlation
		
	return total_kt/num_voters


def sliding_kt_distance(t, window_size):
	kt_dict = {}
	for test in preferences.keys():
		# print(test)
		kt_dict[test] = []
		avg_run_kt = [0]*window_size
		kt = 0
		for run in preferences[test].keys():
			curr_pref = preferences[test][run][t - window_size : t]
			kt_iters = []
			for pref in curr_pref:
				kt_iters.append(kt_distance(preferences[test][run][t], pref, num_voters))
			for i in range(window_size):
				avg_run_kt[i] += kt_iters[i]        # sum over different runs

		for i in range(window_size):
			avg_run_kt[i] /= avg_runs
		kt_dict[test] = sum(avg_run_kt) / window_size       # get sum over the window
		# kt_dict[test] = max(avg_run_kt) / window_size     # get max over the window
	return kt_dict


for i in range(iterations - 1, end_interval, -window_size):
	print(i)
	kt_dict = sliding_kt_distance(i, window_size)
	if not kt_dict_interval:
		for test in kt_dict.keys():
			kt_dict_interval[test] = [kt_dict[test]]
	else:
		for test in kt_dict.keys():
			kt_dict_interval[test].append(kt_dict[test])
	

plot = plt.figure()

for test in kt_dict_interval.keys():
    plt.plot(range(iterations - 1, end_interval, -window_size), kt_dict_interval[test], label=test)

curr_dir = os.path.dirname(os.getcwd())
os.makedirs(curr_dir + "/numerical_results/" + voting_rule, exist_ok=True)
os.makedirs(curr_dir + "/graph_results/" + voting_rule, exist_ok=True)

plt.legend(loc='upper right')
plt.xlabel("Number of iterations")
plt.ylabel("Avg kendalltau distance - ")
plt.show()
graph_file = 'kt_distance_setting_{}_{}_voter_'.format(voting_setting, voting_rule) + str(num_voters) + '_cand_' + str(num_candidates) + \
                "_committee_size_" + str(committee_size) + "_iter_" + str(iterations) + "_to_" + str(end_interval) + "_avg_" + str(avg_runs) + "_window_size_" + str(window_size) + '.png'
graph_path = os.path.join(curr_dir + "/graph_results/" + voting_rule + "/", graph_file)

plt.savefig(graph_path)



# get kt dist over 0-50000 iters, tie breaking using disctioanry order
# July 29 - 14, next meet 16