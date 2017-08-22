import os
import sys

import pickle

import train
import utils
import random
import configparser
import dataset as ds
import distutils.util
import tensorflow as tf
from pprint import pprint

import utils_nlp
from langdetect import detect
import nltk
import argparse


from entity_lstm import EntityLSTM


def load_parameters(parameters_filepath=os.path.join('.', 'parameters.ini'), verbose=True):
    '''
    Load parameters from the ini file, and ensure that each parameter is cast to the correct type
    '''
    conf_parameters = configparser.ConfigParser()
    conf_parameters.read(parameters_filepath)
    nested_parameters = utils.convert_configparser_to_dictionary(conf_parameters)
    parameters = {}
    for k, v in nested_parameters.items():
        parameters.update(v)
    for k, v in parameters.items():
        # If the value is a list delimited with a comma, choose one element at random.
        if ',' in v:
            v = random.choice(v.split(','))
            parameters[k] = v
        # Ensure that each parameter is cast to the correct type
        if k in ['character_embedding_dimension', 'character_lstm_hidden_state_dimension', 'token_embedding_dimension',
                 'token_lstm_hidden_state_dimension', 'patience', 'maximum_number_of_epochs', 'maximum_training_time',
                 'number_of_cpu_threads', 'number_of_gpus',
                 'character_hidden_layer', 'token_hidden_layer', 'embedding_dimension', 'batch_size']:
            parameters[k] = int(v)
        elif k in ['dropout_rate', 'learning_rate', 'gradient_clipping_value']:
            parameters[k] = float(v)
        elif k in ['remap_unknown_tokens_to_unk', 'use_character_lstm', 'use_crf', 'train_model',
                   'use_pretrained_model', 'debug', 'verbose',
                   'reload_character_embeddings', 'reload_character_lstm', 'reload_token_embeddings',
                   'reload_token_lstm', 'reload_feedforward', 'reload_crf',
                   'check_for_lowercase', 'check_for_digits_replaced_with_zeros', 'freeze_token_embeddings',
                   'load_only_pretrained_token_embeddings',
                   'gru_neuron', 'use_gazetteer']:
            parameters[k] = distutils.util.strtobool(v)
    if verbose: pprint(parameters)
    return parameters, conf_parameters


def get_valid_dataset_filepaths(parameters):
    dataset_filepaths = {}

    dataset_type = 'predict'
    dataset_filepaths[dataset_type] = os.path.join(parameters['dataset_' + dataset_type])
    return dataset_filepaths


def check_parameter_compatiblity(parameters, dataset_filepaths):
    # Check mode of operation
    if parameters['train_model']:
        if 'train' not in dataset_filepaths or 'valid' not in dataset_filepaths:
            raise IOError(
                "If train_model is set to True, both train and valid set must exist in the specified dataset folder: {0}".format(
                    parameters['dataset_text_folder']))
    elif parameters['use_pretrained_model']:
        if 'train' in dataset_filepaths and 'valid' in dataset_filepaths:
            print(
                "WARNING: train and valid set exist in the specified dataset folder, but train_model is set to FALSE: {0}".format(
                    parameters['dataset_text_folder']))
        if 'test' not in dataset_filepaths and 'deploy' not in dataset_filepaths:
            raise IOError(
                "For prediction mode, either test set and deploy set must exist in the specified dataset folder: {0}".format(
                    parameters['dataset_text_folder']))
    else:
        raise ValueError('At least one of train_model and use_pretrained_model must be set to True.')

    if parameters['use_pretrained_model']:
        if all([not parameters[s] for s in
                ['reload_character_embeddings', 'reload_character_lstm', 'reload_token_embeddings', 'reload_token_lstm',
                 'reload_feedforward', 'reload_crf']]):
            raise ValueError(
                'If use_pretrained_model is set to True, at least one of reload_character_embeddings, reload_character_lstm, reload_token_embeddings, reload_token_lstm, reload_feedforward, reload_crf must be set to True.')

    if parameters['gradient_clipping_value'] < 0:
        parameters['gradient_clipping_value'] = abs(parameters['gradient_clipping_value'])

def detect_language(sentence):
    """
    detect the langage among FR, ENG and GER, if it's none of them, we assume that the langage was english
    :param sentence: sentence to detect the langage from
    :type sentence: str
    :return: en, fr, de
    :rtype: str
    """
    language = detect(sentence)
    supported_langage = ['fr', 'en', 'de']
    return language if language in supported_langage else 'en'


def process_input(parameters):
    english_model = "../output/en_combined_wp3_0.1_2017-08-14_23-28-52-854871/model/model_001.009.ckpt"
    german_model = "../output/de_combined_wp3_0.2_2017-08-15_08-24-48-871337/model/model_001.039.ckpt"
    with open(parameters['predict_text'], "r", encoding="utf-8") as file:
        text = file.read()

    language = detect_language(text)
    parameters["dataset_predict"] =parse_data(text)
    parameters["language"] = language
    parameters["train_mode"] = False
    parameters["use_pretrained_model"] = True
    parameters["pretrained_model_name"] = "demo"
    print(language)
    if language == "de":
        parameters['pretrained_model_checkpoint_filepath'] = german_model
    else:
        parameters['pretrained_model_checkpoint_filepath'] = english_model
    return parameters

def parse_data(text):
    parsed_data = nltk.sent_tokenize(text)
    output_file = "data_for_model.txt"
    with(open(output_file, "w", encoding="utf-8")) as file:
        for sentence in parsed_data:
            tokens = nltk.word_tokenize(sentence)
            for t in tokens:
                file.write("{0} X O\n".format(t, "X"))
            file.write("\n")
    return output_file


def main(args):
    parameters, conf_parameters = load_parameters()
    if args.file:
        parameters['predict_text'] = args.file
    parameters = process_input(parameters)

    if not parameters["use_pretrained_model"]:
        raise IOError('Set use_pretrained_model parameter to True if you want to predict')
    dataset_filepaths = get_valid_dataset_filepaths(parameters)
    #check_parameter_compatiblity(parameters, dataset_filepaths)



    # Load dataset
    #dataset = ds.Dataset(verbose=parameters['verbose'], debug=parameters['debug'])
    #dataset.load_vocab_word_embeddings(parameters)

    pretrained_model_folder = os.path.dirname(parameters['pretrained_model_checkpoint_filepath'])
    dataset = pickle.load(open(os.path.join(pretrained_model_folder, 'dataset.pickle'), 'rb'))
    #dataset.load_dataset(dataset_filepaths, parameters)
    dataset_type = "predict"
    dataset.labels[dataset_type], dataset.tokens[dataset_type], _, _, _  = dataset._parse_dataset(dataset_filepaths.get(dataset_type, None), parameters['language'])
    #dataset.load_vocab_word_embeddings(parameters)
    iteration_number = 0
    dataset.token_to_index = dict()
    for token_sentence in dataset.tokens['predict']:
        for token in token_sentence:
            if iteration_number == dataset.UNK_TOKEN_INDEX: iteration_number += 1
            if iteration_number == dataset.PADDING_TOKEN_INDEX: iteration_number += 1

            if not utils_nlp.is_token_in_pretrained_embeddings(token, dataset.vocab_embeddings, parameters):
                if parameters['embedding_type'] == 'glove':
                    dataset.token_to_index[token] =  dataset.UNK_TOKEN_INDEX
                    dataset.number_of_unknown_tokens += 1
                    dataset.tokens_mapped_to_unk.append(token)
                elif parameters['embedding_type'] == 'fasttext':
                    dataset.token_to_index[token] = iteration_number
                    iteration_number += 1
                else:
                    raise AssertionError("Embedding type not recognized")
            else:
                if token not in dataset.token_to_index:
                    dataset.token_to_index[token] = iteration_number
                    iteration_number += 1

    dataset_type = "predict"
    for dataset_type in dataset_filepaths.keys():
        dataset.token_indices[dataset_type] = []
        dataset.characters[dataset_type] = []
        dataset.character_indices[dataset_type] = []
        dataset.token_lengths[dataset_type] = []
        dataset.sequence_lengths[dataset_type] = []
        dataset.longest_token_length_in_sequence[dataset_type] = []
        # character_indices_padded[dataset_type] = []
        for token_sequence in dataset.tokens[dataset_type]:
            dataset.token_indices[dataset_type].append([dataset.token_to_index.get(token, dataset.UNK_TOKEN_INDEX) for token in token_sequence])
            dataset.characters[dataset_type].append([list(token) for token in token_sequence])
            dataset.character_indices[dataset_type].append(
                [[dataset.character_to_index.get(character,dataset.UNK_CHARACTER_INDEX) for character in token] for token in token_sequence])
            dataset.token_lengths[dataset_type].append([len(token) for token in token_sequence])
            dataset.sequence_lengths[dataset_type].append(len(token_sequence))
            dataset.longest_token_length_in_sequence[dataset_type].append(max(dataset.token_lengths[dataset_type][-1]))

            # character_indices_padded[dataset_type].append([ utils.pad_list(temp_token_indices, longest_token_length_in_sequence, self.PADDING_CHARACTER_INDEX)
            #                                                for temp_token_indices in character_indices[dataset_type][-1]])

        dataset.label_indices[dataset_type] = []
        for label_sequence in dataset.labels[dataset_type]:
            dataset.label_indices[dataset_type].append([dataset.label_to_index[label] for label in label_sequence])

    tmp_vector = [0] * len(dataset.unique_labels)
    tmp_vector[dataset.label_to_index["O"]] = 1
    dataset.PADDING_LABEL_VECTOR = tmp_vector
    for dataset_type in dataset_filepaths.keys():
        dataset.label_vector_indices[dataset_type] = []
        for label_indices_sequence in dataset.label_indices[dataset_type]:
            vector_sequence = []
            for indice in label_indices_sequence:
                vector = [0] * len(dataset.unique_labels)
                vector[indice] = 1
                vector_sequence.append(vector)
            dataset.label_vector_indices[dataset_type].append(vector_sequence)

    # Create graph and session
    with tf.Graph().as_default():
        session_conf = tf.ConfigProto(
            intra_op_parallelism_threads=parameters['number_of_cpu_threads'],
            inter_op_parallelism_threads=parameters['number_of_cpu_threads'],
            device_count={'CPU': 1, 'GPU': parameters['number_of_gpus']},
            allow_soft_placement=True,
            # automatically choose an existing and supported device to run the operations in case the specified one doesn't exist
            log_device_placement=False,
        )
        session_conf.gpu_options.allow_growth = True


        sess = tf.Session(config=session_conf)

        model = EntityLSTM(dataset, parameters)
        model_saver = tf.train.Saver()

        prediction_folder = os.path.join('..', 'predictions')
        utils.create_folder_if_not_exists(prediction_folder)
        dataset_name = parameters['pretrained_model_name']
        model_name = '{0}_{1}'.format(parameters["language"] + "_" + dataset_name,
                                      utils.get_current_time_in_miliseconds())
        prediction_folder = os.path.join(prediction_folder, model_name)
        utils.create_folder_if_not_exists(prediction_folder)
        epoch_number = 100
        #dataset_name = utils.get_basename_without_extension(parameters['dataset_test'])
        with open(os.path.join(prediction_folder, 'parameters.ini'), 'w') as parameters_file:
            conf_parameters.write(parameters_file)



        if parameters['use_pretrained_model']:
            # Restore pretrained model parameters
            transition_params_trained = train.restore_model_parameters_from_pretrained_model(parameters, dataset, sess, model, model_saver)
            model.load_pretrained_token_embeddings(sess, dataset, parameters)

            demo = parameters['pretrained_model_name'] == "demo"
            y_pred, y_true, output_filepaths = train.predict_labels(sess, model, transition_params_trained, parameters, dataset, epoch_number, prediction_folder, dataset_filepaths, demo=demo)

            if parameters['pretrained_model_name'] == "demo":
                print("============")
                print(" Prediction ")
                print("============")
                i = 0
                for sentence in dataset.tokens['predict']:
                    for token in sentence:
                        predict_label = dataset.index_to_label[y_pred['predict'][i]]
                        if dataset.index_to_label[y_pred['predict'][i]] != "O":
                            print(token,predict_label)
                        else:
                            print(token)
                        i += 1
                    print("")
        else:
            raise IOError('Set use_pretrained_model parameter to True')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--file', type=str,
                        help='input file of the demo')
    args = parser.parse_args()
    main(args)
