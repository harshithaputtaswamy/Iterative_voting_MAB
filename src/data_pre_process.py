import json
from preflibtools.properties import borda_scores, has_condorcet
from preflibtools.instances import OrdinalInstance


def process_soc_data(input_data_file, output_data_file):
    instance = OrdinalInstance()
    instance.parse_file(input_data_file)

    flattened_res = instance.flatten_strict()
    res_dict = dict()

    for res in flattened_res:
        res_dict[json.dumps(res[0])] = res[1]

    with open(output_data_file, "w") as outfile:
        json.dump(res_dict, outfile)

    print(borda_scores(instance), has_condorcet(instance))

process_soc_data("../datasets/soc/00004-00000024.soc", "parsed_soc_data.json")


def generate_IC_data(output_data_file):
    instance = OrdinalInstance()
    instance.populate_IC(10, 3)

    flattened_res = instance.flatten_strict()
    res_dict = dict()

    for res in flattened_res:
        res_dict[json.dumps(res[0])] = res[1]

    with open(output_data_file, "w") as outfile:
        json.dump(res_dict, outfile)

    print(borda_scores(instance), has_condorcet(instance))
generate_IC_data("IC_data.json")
