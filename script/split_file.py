import argparse
import random
import numpy as np
from tqdm import tqdm
def split_file(file_path):
    sentences = []
    tmp = []
    with open(file_path, "r", encoding="utf-8") as file:
        for line in tqdm(file):
            if line.strip():
                tmp.append(line)
            elif tmp:
                tmp.append("")
                sentences.append("".join(tmp))
                tmp = []
            else:
                tmp = []

    args =list(range(0, len(sentences)))
    random.shuffle(args)
    args_test = args[0:4000]
    args_valid = args[4000:8000]
    args_train = args[8000:]

    save_file("\n".join(np.array(sentences)[args_test]), file_path+".test")
    print("test saved")
    save_file("\n".join(np.array(sentences)[args_valid]), file_path+".valid")
    print("valid saved")
    save_file("\n".join(np.array(sentences)[args]), file_path+".train")
    print("train saved")


def save_file(data, file_path):
    with open(file_path, "w", encoding="utf-8") as file:
        file.write(data)



if __name__ == "__main__":
    #argparser = argparse.ArgumentParser()
    #argparser.add_argument("--file", type=str, help="file to be splitted",
     #                        required=False)
    #parsed_args = argparser.parse_args()
    #split_file(parsed_args.file)
    file_path = "aij-wikiner-de-wp2-simplified"
    split_file(file_path)