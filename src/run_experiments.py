'''
To run:
CUDA_VISIBLE_DEVICES="" python3.5 main.py &
CUDA_VISIBLE_DEVICES=1 python3.5 main.py &
CUDA_VISIBLE_DEVICES=2 python3.5 main.py &
CUDA_VISIBLE_DEVICES=3 python3.5 main.py &
'''
from __future__ import print_function

import glob
import json
import numpy as np
import os
import pickle
import shutil
import tensorflow as tf
from tqdm import tqdm

import utils

np.random.seed(0)
import matplotlib
import copy
import distutils.util
import codecs

matplotlib.use('Agg')
import dataset as ds
import time
import random
import sys

random.seed(0)
import evaluate
import configparser
import train
import logging
import argparse
import constants

from pprint import pprint
from entity_lstm import EntityLSTM
from tensorflow.contrib.tensorboard.plugins import projector
from itertools import product
from utils import keep_only_best_model, create_folder_if_not_exists

# http://stackoverflow.com/questions/42217532/tensorflow-version-1-0-0-rc2-on-windows-opkernel-op-bestsplits-device-typ
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
print('NeuroNER version: {0}'.format('1.0-dev'))
print('TensorFlow version: {0}'.format(tf.__version__))

import warnings

warnings.filterwarnings('ignore')


def load_parameters(parameters_filepath=os.path.join('.', 'parameters.ini'), verbose=True):
    '''
    Load parameters from the ini file, and ensure that each parameter is cast to the correct type
    '''
    conf_parameters = configparser.ConfigParser()
    conf_parameters.read(parameters_filepath)
    return conf_parameters

def parse_parameters(conf_parameters):
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
                 'character_hidden_layer', 'token_hidden_layer', 'embedding_dimension', 'batch_size',
                 'cross_validation', 'data_to_use']:
            if k in parameters:
                parameters[k] = int(v)
        elif k in ['dropout_rate', 'learning_rate', 'gradient_clipping_value']:
            parameters[k] = float(v)
        elif k in ['remap_unknown_tokens_to_unk', 'use_character_lstm', 'use_crf', 'train_model',
                   'use_pretrained_model', 'debug', 'verbose',
                   'reload_character_embeddings', 'reload_character_lstm', 'reload_token_embeddings',
                   'reload_token_lstm', 'reload_feedforward', 'reload_crf',
                   'check_for_lowercase', 'check_for_digits_replaced_with_zeros', 'freeze_token_embeddings',
                   'load_only_pretrained_token_embeddings',
                   'gru_neuron']:
            parameters[k] = distutils.util.strtobool(v)
    # if verbose: pprint(parameters)
    return parameters, conf_parameters


def get_valid_dataset_filepaths(parameters):
    dataset_filepaths = {}
    if 'cross_validation' in parameters and parameters['cross_validation']:
        dataset_filepaths['train'] = os.path.join(parameters['dataset_train'])
    else:
        for dataset_type in ['train', 'valid', 'test']:
            dataset_filepaths[dataset_type] = os.path.join(parameters['dataset_' + dataset_type])
    return dataset_filepaths


def check_parameter_compatiblity(parameters, dataset_filepaths):
    # Check mode of operation
    if parameters['train_model']:
        if 'train' not in dataset_filepaths:  # or 'valid' not in dataset_filepaths:
            raise IOError(
                "If train_model is set to True, both train and valid set must exist in the specified dataset folder: {0}".format(
                    parameters['dataset_text_folder']))
    elif parameters['use_pretrained_model']:
        if 'test' not in dataset_filepaths and 'deploy' not in dataset_filepaths:
            raise IOError(
                "For prediction mode, either test set and deploy set must exist in the specified dataset folder: {0}".format(
                    parameters['dataset_text_folder']))
    else:  # if not parameters['train_model'] and not parameters['use_pretrained_model']:
        raise ValueError('At least one of train_model and use_pretrained_model must be set to True.')

    if parameters['use_pretrained_model']:
        if all([not parameters[s] for s in
                ['reload_character_embeddings', 'reload_character_lstm', 'reload_token_embeddings', 'reload_token_lstm',
                 'reload_feedforward', 'reload_crf']]):
            raise ValueError(
                'If use_pretrained_model is set to True, at least one of reload_character_embeddings, reload_character_lstm, reload_token_embeddings, reload_token_lstm, reload_feedforward, reload_crf must be set to True.')

    if parameters['gradient_clipping_value'] < 0:
        parameters['gradient_clipping_value'] = abs(parameters['gradient_clipping_value'])


def set_datasets(conf_parameters, language):
    if language not in ['en','fr','it','de']:
        file_name = "../../../new_dataset/{}/wp3/combined_wp3_1.0".format(language)
        conf_parameters.set('dataset', 'dataset_train', "{}.train".format(file_name))
        conf_parameters.set('dataset', 'dataset_valid', "{}.valid_p".format(file_name))
        conf_parameters.set('dataset', 'dataset_test', "{}.test_p".format(file_name))
    else:
        file_name = "../../../new_dataset/{}/wp3/combined_wp3_0.3".format(language)
        conf_parameters.set('dataset', 'dataset_train', "{}.train".format(file_name))
        conf_parameters.set('dataset', 'dataset_valid', "{}.valid".format(file_name))
        conf_parameters.set('dataset', 'dataset_test', "{}.test".format(file_name))


        #parameters['dataset_train'] = "{}.train".format(file_name)
        #parameters['dataset_valid'] = "{}.valid".format(file_name)
        #parameters['dataset_test'] = "{}.test".format(file_name)
    return conf_parameters


def main(languages):
    #embeddings_type = ['polyglot', 'fasttext']
    #embeddings_type = ['fasttext', 'fasttext_noOOV']
    embeddings_type = ['fasttext_noOOV']
    character_lstm = [True]
    embedding_language = ['target', 'source']
    combination = product(languages, embeddings_type, embedding_language, character_lstm)
    create_folder_if_not_exists(os.path.join("..", "log"))
    experiment_timestamp = utils.get_current_time_in_miliseconds()
    log_file = os.path.join("..", "log", "experiment-{}.log".format(experiment_timestamp))

    for language, emb_type, emb_language, char_lstm in combination:
        conf_parameters = load_parameters()
        conf_parameters = set_datasets(conf_parameters, language)
        conf_parameters.set('ann','use_character_lstm', str(char_lstm))
        conf_parameters.set('ann','embedding_type', emb_type)
        conf_parameters.set('ann','embedding_language', emb_language)
        if emb_type == 'polyglot':
            conf_parameters.set('ann', 'embedding_dimension', str(64))
        elif 'fasttext' in emb_type:
            conf_parameters.set('ann', 'embedding_dimension', str(300))
        else:
            raise("Uknown embedding type")
        if emb_language == 'source':
            conf_parameters.set('dataset', 'language', constants.MAPPING_LANGUAGE[language])
        else:
            conf_parameters.set('dataset', 'language', language)
        parameters, conf_parameters = parse_parameters(conf_parameters)

        start_time = time.time()
        experiment_timestamp = utils.get_current_time_in_miliseconds()

        results = {}
        results['epoch'] = {}
        results['execution_details'] = {}
        results['execution_details']['train_start'] = start_time
        results['execution_details']['time_stamp'] = experiment_timestamp
        results['execution_details']['early_stop'] = False
        results['execution_details']['keyboard_interrupt'] = False
        results['execution_details']['num_epochs'] = 0
        results['model_options'] = copy.copy(parameters)

        dataset_name = utils.get_basename_without_extension(parameters['dataset_train'])
        model_name = '{0}_{1}_{2}_{3}_{4}'.format(language, emb_type, char_lstm, emb_language,
                                                  results['execution_details']['time_stamp'])

        sys.stdout = open(os.path.join("..", "log", model_name), "w")
        print(language, emb_type, char_lstm, emb_language)

        with open(log_file, "a") as file:
            file.write("Experiment: {}\n".format(model_name))
            file.write("Start time:{}\n".format(experiment_timestamp))
            file.write("-------------------------------------\n\n")
        pprint(parameters)
        dataset_filepaths = get_valid_dataset_filepaths(parameters)
        check_parameter_compatiblity(parameters, dataset_filepaths)
        previous_best_valid_epoch = -1

        # Load dataset
        dataset = ds.Dataset(verbose=parameters['verbose'], debug=parameters['debug'])
        dataset.load_vocab_word_embeddings(parameters)
        dataset.load_dataset(dataset_filepaths, parameters)

        # Create graph and session
        with tf.Graph().as_default():
            session_conf = tf.ConfigProto(
                intra_op_parallelism_threads=parameters['number_of_cpu_threads'],
                inter_op_parallelism_threads=parameters['number_of_cpu_threads'],
                device_count={'CPU': 1, 'GPU': parameters['number_of_gpus']},
                allow_soft_placement=True,
                # automatically choose an existing and supported device to run the operations in case the specified one doesn't exist
                log_device_placement=False
            )

            session_conf.gpu_options.allow_growth = True

            sess = tf.Session(config=session_conf)

            with sess.as_default():
                # Initialize and save execution details

                print(model_name)
                output_folder = os.path.join('..', 'output')
                utils.create_folder_if_not_exists(output_folder)
                stats_graph_folder = os.path.join(output_folder, model_name)  # Folder where to save graphs
                utils.create_folder_if_not_exists(stats_graph_folder)
                model_folder = os.path.join(stats_graph_folder, 'model')
                utils.create_folder_if_not_exists(model_folder)
                with open(os.path.join(model_folder, 'parameters.ini'), 'w') as parameters_file:
                    conf_parameters.write(parameters_file)
                tensorboard_log_folder = os.path.join(stats_graph_folder, 'tensorboard_logs')
                utils.create_folder_if_not_exists(tensorboard_log_folder)
                tensorboard_log_folders = {}
                for dataset_type in dataset_filepaths.keys():
                    tensorboard_log_folders[dataset_type] = os.path.join(stats_graph_folder, 'tensorboard_logs',
                                                                         dataset_type)
                    utils.create_folder_if_not_exists(tensorboard_log_folders[dataset_type])
                # del dataset.embeddings_matrix
                if not parameters['use_pretrained_model']:
                    pickle.dump(dataset, open(os.path.join(model_folder, 'dataset.pickle'), 'wb'))
                # dataset.load_pretrained_word_embeddings(parameters)
                # Instantiate the model
                # graph initialization should be before FileWriter, otherwise the graph will not appear in TensorBoard
                model = EntityLSTM(dataset, parameters)

                # Instantiate the writers for TensorBoard
                writers = {}
                for dataset_type in dataset_filepaths.keys():
                    writers[dataset_type] = tf.summary.FileWriter(tensorboard_log_folders[dataset_type],
                                                                  graph=sess.graph)
                embedding_writer = tf.summary.FileWriter(
                    model_folder)  # embedding_writer has to write in model_folder, otherwise TensorBoard won't be able to view embeddings

                embeddings_projector_config = projector.ProjectorConfig()
                tensorboard_token_embeddings = embeddings_projector_config.embeddings.add()
                tensorboard_token_embeddings.tensor_name = model.token_embedding_weights.name
                token_list_file_path = os.path.join(model_folder, 'tensorboard_metadata_tokens.tsv')
                tensorboard_token_embeddings.metadata_path = os.path.relpath(token_list_file_path, '..')

                if parameters['use_character_lstm']:
                    tensorboard_character_embeddings = embeddings_projector_config.embeddings.add()
                    tensorboard_character_embeddings.tensor_name = model.character_embedding_weights.name
                    character_list_file_path = os.path.join(model_folder, 'tensorboard_metadata_characters.tsv')
                    tensorboard_character_embeddings.metadata_path = os.path.relpath(character_list_file_path, '..')

                projector.visualize_embeddings(embedding_writer, embeddings_projector_config)

                # Write metadata for TensorBoard embeddings
                token_list_file = codecs.open(token_list_file_path, 'w', 'UTF-8')
                for token_index in range(len(dataset.index_to_token)):
                    token_list_file.write('{0}\n'.format(dataset.index_to_token[token_index]))
                token_list_file.close()

                if parameters['use_character_lstm']:
                    character_list_file = codecs.open(character_list_file_path, 'w', 'UTF-8')
                    for character_index in range(dataset.alphabet_size):
                        if character_index == dataset.PADDING_CHARACTER_INDEX:
                            character_list_file.write('PADDING\n')
                        else:
                            character_list_file.write('{0}\n'.format(dataset.index_to_character[character_index]))
                    character_list_file.close()

                try:
                    # Initialize the model
                    sess.run(tf.global_variables_initializer())
                    if not parameters['use_pretrained_model']:
                        model.load_pretrained_token_embeddings(sess, dataset, parameters)

                    # Start training + evaluation loop. Each iteration corresponds to 1 epoch.
                    bad_counter = 0  # number of epochs with no improvement on the validation test in terms of F1-score
                    previous_best_valid_f1_score = -1
                    transition_params_trained = np.random.rand(len(dataset.unique_labels), len(
                        dataset.unique_labels))  # TODO np.random.rand(len(dataset.unique_labels)+2,len(dataset.unique_labels)+2)
                    model_saver = tf.train.Saver(
                        max_to_keep=None)  # parameters['maximum_number_of_epochs'])  # defaults to saving all variables
                    epoch_number = 0

                    while True:
                        step = 0
                        epoch_number += 1
                        print('\nStarting epoch {0}'.format(epoch_number))

                        epoch_start_time = time.time()

                        if parameters['use_pretrained_model'] and epoch_number == 1:
                            # Restore pretrained model parameters
                            transition_params_trained = train.restore_model_parameters_from_pretrained_model(parameters,
                                                                                                             dataset,
                                                                                                             sess,
                                                                                                             model,
                                                                                                             model_saver)
                        elif epoch_number != 0:
                            # Train model: loop over all sequences of training set with shuffling
                            sequence_numbers = list(range(len(dataset.token_indices['train'])))
                            random.shuffle(sequence_numbers)
                            data_counter = 0
                            sub_id = 0
                            for i in tqdm(range(0, len(sequence_numbers), parameters['batch_size']), "Training epoch {}".format(epoch_number),
                                          mininterval=1):
                                data_counter += parameters['batch_size']
                                if data_counter >= 20000:
                                    data_counter = 0
                                    sub_id += 0.001
                                    print("Intermediate evaluation number: ", sub_id)
                                    epoch_elapsed_training_time = time.time() - epoch_start_time
                                    print('Training completed in {0:.2f} seconds'.format(epoch_elapsed_training_time),
                                          flush=True)

                                    y_pred, y_true, output_filepaths = train.predict_labels(sess, model,
                                                                                            transition_params_trained,
                                                                                            parameters, dataset,
                                                                                            epoch_number + sub_id,
                                                                                            stats_graph_folder,
                                                                                            dataset_filepaths)
                                    # Evaluate model: save and plot results
                                    evaluate.evaluate_model(results, dataset, y_pred, y_true, stats_graph_folder,
                                                            epoch_number, epoch_start_time, output_filepaths,
                                                            parameters)
                                    # Save model
                                    model_saver.save(sess, os.path.join(model_folder,
                                                                        'model_{0:07.3f}.ckpt'.format(
                                                                            epoch_number + sub_id)))
                                    # Save TensorBoard logs
                                    summary = sess.run(model.summary_op, feed_dict=None)
                                    writers['train'].add_summary(summary, epoch_number)
                                    writers['train'].flush()
                                    utils.copytree(writers['train'].get_logdir(), model_folder)
                                    # Early stop
                                    valid_f1_score = results['epoch'][epoch_number][0]['valid']['f1_score']['micro']
                                    if valid_f1_score > previous_best_valid_f1_score:
                                        bad_counter = 0
                                        previous_best_valid_f1_score = valid_f1_score
                                    else:
                                        bad_counter += 1

                                sequence_number = sequence_numbers[i: i + parameters['batch_size']]
                                transition_params_trained, loss = train.train_step(sess, dataset, sequence_number,
                                                                                   model, transition_params_trained,
                                                                                   parameters)
                        epoch_elapsed_training_time = time.time() - epoch_start_time
                        print('Training completed in {0:.2f} seconds'.format(epoch_elapsed_training_time), flush=True)

                        y_pred, y_true, output_filepaths = train.predict_labels(sess, model, transition_params_trained,
                                                                                parameters, dataset, epoch_number,
                                                                                stats_graph_folder, dataset_filepaths)

                        # Evaluate model: save and plot results
                        evaluate.evaluate_model(results, dataset, y_pred, y_true, stats_graph_folder, epoch_number,
                                                epoch_start_time, output_filepaths, parameters)

                        # Save model
                        model_saver.save(sess, os.path.join(model_folder, 'model_{0:05d}.ckpt'.format(epoch_number)))

                        # Save TensorBoard logs
                        summary = sess.run(model.summary_op, feed_dict=None)
                        writers['train'].add_summary(summary, epoch_number)
                        writers['train'].flush()
                        utils.copytree(writers['train'].get_logdir(), model_folder)

                        # Early stop
                        valid_f1_score = results['epoch'][epoch_number][0]['valid']['f1_score']['micro']
                        if valid_f1_score > previous_best_valid_f1_score:
                            bad_counter = 0
                            previous_best_valid_f1_score = valid_f1_score
                            previous_best_valid_epoch = epoch_number
                        else:
                            bad_counter += 1
                        print("The last {0} epochs have not shown improvements on the validation set.".format(
                            bad_counter))

                        if bad_counter >= parameters['patience']:
                            print('Early Stop!')
                            results['execution_details']['early_stop'] = True
                            break

                        if epoch_number >= parameters['maximum_number_of_epochs']: break

                    keep_only_best_model(model_folder,previous_best_valid_epoch ,parameters['maximum_number_of_epochs']+1)

                except KeyboardInterrupt:
                    results['execution_details']['keyboard_interrupt'] = True
                    print('Training interrupted')
                    # remove the experiment
                    remove_experiment = input("Do you want to remove the experiment? (yes/y/Yes)")
                    if remove_experiment in ["Yes", "yes", "y"]:
                        shutil.rmtree(stats_graph_folder)
                        print("Folder removed")
                    else:
                        print('Finishing the experiment')
                        end_time = time.time()
                        results['execution_details']['train_duration'] = end_time - start_time
                        results['execution_details']['train_end'] = end_time
                        evaluate.save_results(results, stats_graph_folder)
                    sys.stdout.close()
                except Exception:
                    logging.exception("")
                    remove_experiment = input("Do you want to remove the experiment? (yes/y/Yes)")
                    if remove_experiment in ["Yes", "yes", "y"]:
                        shutil.rmtree(stats_graph_folder)
                        print("Folder removed")
                    sys.stdout.close()

            sess.close()  # release the session's resources
            sys.stdout.close()
if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--languages", nargs='+', type=str,
                           help="list of names of the configuration in the parameter file",
                           required=True)
    parsed_args = argparser.parse_args()

    main(parsed_args.languages)


