import json

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
