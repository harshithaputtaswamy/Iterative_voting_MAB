import os
import json
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



voting_rule = "pav"
curr_dir = os.path.dirname(os.getcwd())
file_to_read = "setting_1_pav_tie_breaking_rand_voter_10_cand_5_committee_size_3_iter_50000_avg_50.json"
output_path = os.path.join(curr_dir + "/numerical_results/" + voting_rule + "/", file_to_read)
print(output_path)


read_result = read_results(output_path)
run_setting = read_result.get_run_setting()
preferences = read_result.get_preferences()
rewards = read_result.get_rewards()

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
		if voting_rule == "plurality" or voting_rule == 'anti_plurality':
			total_kt += 0 if preference_dict_t[str(voter)] != preference_dict_t_n[str(voter)] else 1
		else:
			res = stats.kendalltau(preference_dict_t[str(voter)], preference_dict_t_n[str(voter)])
			total_kt += res.statistic       # res.statistic closer to 1 then it implies strong positive 
			# ordinal correlation, 0 implies weak ordinal correlation, -1 implies strong negative ordinal correlation
	return total_kt/num_voters


def sliding_kt_distance(t):
	kt_dict = {}
	for test in preferences.keys():
		# print(test)
		kt_dict[test] = []
		avg_run_kt = [0]*window_size
		for run in preferences[test].keys():
			curr_pref = preferences[test][run][abs(t - window_size) : t]
			kt_iters = []
			for pref in curr_pref:
				kt_iters.append(kt_distance(preferences[test][run][t-1], pref, num_voters))
			for i in range(window_size):
				avg_run_kt[i] += kt_iters[i]        # sum over different runs

		for i in range(window_size):
			avg_run_kt[i] /= avg_runs
		kt_dict[test] = sum(avg_run_kt) / window_size       # get sum over the window
		# kt_dict[test] = max(avg_run_kt) / window_size     # get max over the window
	return kt_dict


# calculate the average KT distance between preferences of voters in given itervals
for i in range(iterations, end_interval, -window_size):
	print(i)
	kt_dict = sliding_kt_distance(i)
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
plt.ylabel("Avg kendalltau corelation - ")
plt.show()
if voting_rule == "approval":
	graph_file = 'kt_distance_setting_{}_{}_tie_breaking_{}_approval_count_{}_voter_'.format(voting_setting, voting_rule, tie_breaking_rule, approval_count) + str(num_voters) + '_cand_' + str(num_candidates) + \
                "_committee_size_" + str(committee_size) + "_iter_" + str(iterations) + "_to_" + str(end_interval) + "_avg_" + str(avg_runs) + "_window_size_" + str(window_size) + '.png'
else:
	graph_file = 'kt_distance_setting_{}_{}_tie_breaking_{}_voter_'.format(voting_setting, voting_rule, tie_breaking_rule) + str(num_voters) + '_cand_' + str(num_candidates) + \
                "_committee_size_" + str(committee_size) + "_iter_" + str(iterations) + "_to_" + str(end_interval) + "_avg_" + str(avg_runs) + "_window_size_" + str(window_size) + '.png'
graph_path = os.path.join(curr_dir + "/graph_results/" + voting_rule + "/", graph_file)

plt.savefig(graph_path)


def cost_of_strategy():
	cost_dict = {}
	for test in rewards.keys():
		cost_dict[test] = []
		avg_run_cost = [0]*iterations
		for run in rewards[test].keys():
			true_reward = rewards[test][run][0]
			for iter in range(iterations):
				#get the difference between 0th borda score and nth borda score
				avg_run_cost[iter] += (true_reward - rewards[test][run][iter])
		for iter in range(iterations):
			avg_run_cost[iter] /= avg_runs
		cost_dict[test] = avg_run_cost
	return cost_dict


# plot cost of strgategy
cost_of_strategy_dict = cost_of_strategy()

plot = plt.figure()

for test in cost_of_strategy_dict.keys():
    plt.plot(range(iterations), cost_of_strategy_dict[test], label=test)

curr_dir = os.path.dirname(os.getcwd())
os.makedirs(curr_dir + "/numerical_results/" + voting_rule, exist_ok=True)
os.makedirs(curr_dir + "/graph_results/" + voting_rule, exist_ok=True)

plt.legend(loc='upper right')
plt.xlabel("Number of iterations")
plt.ylabel("Avg cost of strategy - ")
plt.show()
if voting_rule == "approval":
	graph_file = 'cost_of_strategy_setting_{}_{}_tie_breaking_{}_approval_count_{}_voter_'.format(voting_setting, voting_rule, tie_breaking_rule, approval_count) + str(num_voters) + '_cand_' + str(num_candidates) + \
                "_committee_size_" + str(committee_size) + "_iter_" + str(iterations) + "_avg_" + str(avg_runs) + '.png'
else:
	graph_file = 'cost_of_strategy_setting_{}_{}_tie_breaking_{}_voter_'.format(voting_setting, voting_rule, tie_breaking_rule) + str(num_voters) + '_cand_' + str(num_candidates) + \
                "_committee_size_" + str(committee_size) + "_iter_" + str(iterations) + "_avg_" + str(avg_runs) + '.png'
graph_path = os.path.join(curr_dir + "/graph_results/" + voting_rule + "/", graph_file)

plt.savefig(graph_path)
