import json
import tqdm
from model import model
from utils import generate_reward

parsed_soc_data = json.load("parsed_soc_data.json")

# train the model for given number of iterations and given training algo
def train(iterations, algo):
    actual_mean = parsed_soc_data["borda_scores"]   #the actual borda score of each candidate
    num_candidates = parsed_soc_data["num_candidates"]
    num_voters = parsed_soc_data["num_voters"]
    full_voting_profile = parsed_soc_data["full_voting_profile"]
    flattened_voting_profile = parsed_soc_data["flattened_voting_profile"]

    # generate borda score dictionary for candidates and initial with 0
    reward = {}
    for cand in range(num_candidates):
        reward[cand] = 0

    agent = model(num_candidates, num_voters, full_voting_profile)

    print("Actual mean ", actual_mean)
    print("before running mean = ", agent.mean_reward)

    # in each iteration the voter will pull a random arm if rand.rand in < epsilon else he will select the canditate in his preference profile
    # with the highest borda score?
    # how to decide which arm is the best arm? something to do with manipulation?
    for i in tqdm(range(iterations)):
        # run one voting cycle where all the voters cast their vote and give the preferences
        for voter in range(num_voters):

            top_candidate, voter_preference = agent.pick_arm(algo, reward, voter)
            reward = generate_reward(top_candidate, voter_preference)

        # update the mean after every voting cycle is done
        agent.update_mean(reward)

