import json
from read_result import get_preferences
from scipy import stats

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


def kt_distance(preference_dict_t, preference_dict_t_n, num_voters):
	total_kt = 0
	for voter in range(num_voters):
		res = stats.kendalltau(preference_dict_t[str(voter)], preference_dict_t_n[str(voter)])
		total_kt += res.statistic
		
	return total_kt/num_voters


def sliding_kt_distance(t, threshold):
	preferences, iterations = get_preferences()
	kt_dict = {}
	for test in preferences.keys():
		print(test)
		kt_dict[test] = {}
		for run in preferences[test].keys():
			kt_dict[test][run] = []
			curr_pref = preferences[test][run]
			curr_pref = curr_pref[iterations - threshold : ]
			# print(curr_pref)
			kt_iters = []
			num_voters = len(curr_pref[0].keys())
			for pref in curr_pref:
				# print(curr_pref[t], pref, num_voters)
				kt_iters.append(kt_distance(curr_pref[t], pref, num_voters))
			kt_dict[test][run] = kt_iters
			# print(kt_iters)
	
	return kt_dict

kt_dict = sliding_kt_distance(-1, 10)
print(kt_dict)
	