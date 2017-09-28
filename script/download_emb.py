import subprocess
import sys
import os
import zipfile
import shutil
from build_embeddings import convert_embeddings


def create_directory_if_not_exist(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def set_up_language(folder, language):
    print("Downloading {}".format(language))
    output_folder = os.path.join(folder, language)

    create_directory_if_not_exist(output_folder)
    command = "wget -P {0} https://s3-us-west-1.amazonaws.com/fasttext-vectors/wiki.{1}.zip".format(output_folder, language)
    subprocess.call(command,shell=True)
    file_path = os.path.join(output_folder,"wiki.{0}.zip".format(language))
    with zipfile.ZipFile(file_path, "r") as zip_ref:
        zip_ref.extractall(output_folder)
    os.remove(file_path)
    convert_embeddings(language, folder)
    vec_file = os.path.join(output_folder,"wiki.{0}.vec".format(language))
    os.remove(vec_file)


if __name__ == '__main__':
    #german
    #languages = ["fr", "it"]
    #languages = ["lb", "nds", "ksh", "pfl", "pdc"]
    languages = ["pms", "lmo", "scn", "vec", "nap", "sc", "co", "rm", "lij", "fur"]
    #french

    folder = "/home/baptiste/projects/word_embeddings/fasttext"
    for language in languages:
        set_up_language(folder, language)
