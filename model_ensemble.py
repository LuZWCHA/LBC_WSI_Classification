import glob
import json
import argparse
import os
from pathlib import Path
import pandas as pd


def check_path(path):
    return os.path.exists(path)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Statistic the detect model output and route it to classify model")
    parser.add_argument(
        '--ensemble_inputs', default="predict result folders of some similar detect models", type=str)
    # parser.add_argument(
    #     '--wsi_input', default="predict result folder of whole image classify model", type=str)
    parser.add_argument(
        '--output', default="cascade model results (of cascade detection models) which format should be CSV", type=str)
    parser.add_argument(
        '--output_mil', default="prepare the input for mil model", type=str)
    parser.add_argument('--origin_input', default="todo", type=str)
    parser.add_argument('--mil_range', default='0.35,0.45',  type=str)

    args = parser.parse_args()

    # read labels and create csv for classify model
    inputs = args.ensemble_inputs
    output = args.output
    origin_input_csv = args.origin_input
    mil_range = args.mil_range.split(",")
    mil_range = [float(x) for x in mil_range]


    inputs = inputs.split(';')

    assert len(inputs) >= 1

    from os.path import join

    file_group = []
    output_dict = []

    empty_flag = False

    first_json_dir = inputs[0]

    filt_names = []
    num_files = -1
    
    origin_image_df = pd.read_csv(origin_input_csv)
    origin_images = list(origin_image_df["slide_path"])
    name2path = {}
    for i in origin_images:
        file_name = Path(i).stem
        name2path[file_name] = i

    print(inputs)
    for idx, input_json_dir in enumerate(inputs[1:]):

        json_files = glob.glob(join(input_json_dir, "*.json"))

        if len(json_files) == 0:
            empty_flag = True
            print("Error: empty folder.")
            break

        json_files.sort()
        file_group.append(json_files)

        if num_files != -1:
            # check the file number, if not same, raise a error.
            assert num_files == len(json_files)
        num_files = len(json_files)

        if idx == 0:
            filt_names = [os.path.basename(f) for f in json_files]

    json_files = glob.glob(join(first_json_dir, "*"))
    json_files = list(filter(lambda a: os.path.basename(a)
                      in filt_names, json_files))
    json_files.sort()
    file_group.append(json_files)
    
    # output dir prepare ...
    parent_dir = os.path.dirname(output)
    if not os.path.exists(parent_dir):
        os.makedirs(parent_dir)

    mil_recheck_csv = args.output_mil
    mil_dict = []
    if not empty_flag:

        for files in zip(*file_group):
            # calculate the averge of the max_scores
            avg_score = 0
            cnt = len(files)
            for json_file in files:
                with open(json_file, 'rb') as f:
                    res = json.load(f)
                max_score = res["max_score"]
                avg_score += max_score

            avg_score /= cnt
            # remove ".json"
            file_name = os.path.basename(json_file)[:-5]
            flag = 1 if avg_score < mil_range[0] or avg_score > mil_range[1] else 0
            
            if flag == 0:
                mil_dict.append((name2path[file_name], os.path.join(first_json_dir, file_name + ".json"), avg_score))
            else:
                output_dict.append((file_name,  avg_score, flag))

        
    print("-" * 100)
    df = pd.DataFrame(output_dict, columns=["slide_name", "max_score", "flag"])
    df.to_csv(output, index=False)
    
    print("-" * 100)
    df = pd.DataFrame(mil_dict, columns=["slide_path", "json_path", "max_score"])
    df.to_csv(mil_recheck_csv, index=False)
    
