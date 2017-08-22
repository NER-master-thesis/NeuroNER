import argparse
import codecs
import random
import numpy as np
from tqdm import tqdm
from glob import glob
import json
import sys



def split_file():
    sentences = []
    tmp = []
    input_file = "../../../new_dataset/als/wp3/combined_processed_wp3_1.0.txt"

    #data = dict()
    #for file in tqdm(input_files):
    #    with open(file, "r") as file:
    #        for line in file:
    #            line = json.loads(line)
    #            if line['text'].strip():
    #                data[line['title']] = line['text']

    with open(input_file, "r", encoding="utf-8") as file:
        for line in tqdm(file):
            if line.strip():
                tmp.append(line)
            elif tmp:
                tmp.append("")
                sentences.append("".join(tmp))
                tmp = []
            else:
                tmp = []


    args = list(range(0, len(sentences)))
    random.shuffle(args)
    output = "../../../new_dataset/als/wp3/combined_processed_wp3_1.0"
    block_size = int(len(args)/5)
    for i in range(0,5):
        args_data = args[i*block_size:(i+1)*block_size]
        save_file("\n".join(np.array(sentences)[args_data]), output + "_"+str(i))
        print("test saved")






"""
    cv = 5
    output_file = "../../../new_dataset/als/wp3/wikipedia_dataset_1.0/combined_wp3_1.0"
    keys = list(data.keys())
    random.shuffle(keys)
    data = {k: data[k] for k in keys}
    dataset_types = list(map(str, list(range(cv))))
    output_data = []
    token_counter = 0
    titles = []
    dataset_type = dataset_types.pop()
    articles_per_block =int(len(data)/cv)
    threshold = 0
    for i, (title, sentences) in enumerate(data.items()):
        if sentences.strip():
            titles.append(title)
            token_counter += len(sentences.strip().split("\n"))
            output_data.append(sentences)
            if token_counter >= threshold + articles_per_block:
                threshold + articles_per_block
                tmp_file = output_file + "_" + dataset_type
                with codecs.open(tmp_file, "w", "utf-8") as file:
                    for line in output_data:
                        file.write(line)
                        file.write("\n")
                tmp_file = tmp_file + "_" + "titles"
                with codecs.open(tmp_file, "w", "utf-8") as file:
                    for t in titles:
                        file.write(t + "\n")
                output_data = []
                titles = []

                token_counter = 0
                if dataset_types:
                    dataset_type = dataset_types.pop()
                    print(dataset_type)
                else:
                    break


"""

def save_file(data, file_path):
    with open(file_path, "w", encoding="utf-8") as file:
        file.write(data)
    print(file_path, "saved")



if __name__ == "__main__":
    #argparser = argparse.ArgumentParser()
    #argparser.add_argument("--file", type=str, help="file to be splitted",
     #                        required=False)
    #parsed_args = argparser.parse_args()
    #split_file(parsed_args.file)
    #file_path = "../"

    split_file()