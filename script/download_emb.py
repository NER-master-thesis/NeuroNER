import subprocess
import sys
import os
import zipfile


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


if __name__ == '__main__':
    #german
    #languages = ["fr", "it"]
    languages = ["it", "fr"]
    #french


    folder = "~/projects/word_embeddings/fasttext"
    for language in languages:
        set_up_language(folder, language)
