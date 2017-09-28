import os
import subprocess


def create_directory_if_not_exist(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

mappging  = {
    'it': 'http://bit.ly/19bShFz'
    #'pms': 'http://bit.ly/12FHY5o',
    #'lmo': 'http://bit.ly/19bSfNO',
    #'scn': 'http://bit.ly/19bTHzV',
    #'vec': 'http://bit.ly/12FJeFu',
    #'rm': 'http://bit.ly/19bSg4h',
}

folder = "/home/baptiste/projects/word_embeddings/polyglot"
for language, url in mappging.items():
    print("Downloading {}".format(language))
    output_folder = os.path.join(folder, language)
    create_directory_if_not_exist(output_folder)
    print(output_folder)
    output_file = os.path.join(output_folder, "polyglot-{}.pkl".format(language))
    print(output_file)
    command = "wget -O {0} {1} ".format(output_file, url)
    subprocess.call(command, shell=True)