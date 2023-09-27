import json
from preflibtools.properties import borda_scores, has_condorcet
from utils import generate_reward 

from preflibtools.instances import OrdinalInstance


# Read from the input .soc file and store it in .json file
def process_soc_data(generate_data, input_data_file = "", output_data_file = ""):
    instance = OrdinalInstance()

    if generate_data:
        instance.populate_IC(10, 3)
    else:
        instance.parse_file(input_data_file)

    res_dict = dict()
    res_dict["num_voters"] = instance.num_voters
    res_dict["num_candidates"] = instance.num_alternatives

    flattened_res = instance.flatten_strict()
    voting_dict = dict()

    for res in flattened_res:
        voting_dict[json.dumps(res[0])] = res[1]

    res_dict["flattened_voting_profile"] = voting_dict
    # res_dict["full_voting_profile"] = [json.loads(key) for key, value in voting_dict.items() for _ in range(value)]
    res_dict["borda_scores"] = borda_scores(instance)

    with open(output_data_file, "w") as outfile:
        json.dump(res_dict, outfile)

    print(borda_scores(instance), generate_reward (voting_dict, res_dict["num_candidates"]))


# process_soc_data(False, "/home/harshitha/fstore/harshitha/soc/00004-00000004.soc", "parsed_soc_data.json")   #to read data from pre existing file
process_soc_data(True, output_data_file = "parsed_soc_data.json")   #to generate data using populate_IC function
