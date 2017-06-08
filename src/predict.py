import os
import sys
import train
import utils
import random
import configparser
import dataset as ds
import distutils.util
import tensorflow as tf
from pprint import pprint
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

    dataset_type = 'test'
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

def main():
    parameters, conf_parameters = load_parameters()
    dataset_filepaths = get_valid_dataset_filepaths(parameters)
    #check_parameter_compatiblity(parameters, dataset_filepaths)

    # Load dataset
    dataset = ds.Dataset(verbose=parameters['verbose'], debug=parameters['debug'])
    dataset.load_pretrained_word_embeddings(parameters)
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

        sess = tf.Session(config=session_conf)

        model = EntityLSTM(dataset, parameters)
        model_saver = tf.train.Saver()

        epoch_number = 100
        dataset_name = utils.get_basename_without_extension(parameters['dataset_test'])
        model_name = 'predictions'
        output_folder = os.path.join('..', 'output')
        prediction_folder = os.path.join(output_folder, model_name)  # Folder where to save graphs
        if not os.path.exists(prediction_folder):
            os.makedirs(prediction_folder)

        if parameters['use_pretrained_model']:
            # Restore pretrained model parameters
            transition_params_trained = train.restore_model_parameters_from_pretrained_model(parameters, dataset, sess, model, model_saver)
            y_pred, y_true, output_filepaths = train.predict_labels(sess, model, transition_params_trained, parameters, dataset, epoch_number, prediction_folder, dataset_filepaths)
            print('True labels')
            print([dataset.index_to_label[p] for p in y_true['test']])
            print('Predicted labels')
            print([dataset.index_to_label[p] for p in y_pred['test']])
        else:
            raise IOError('Set use_pretrained_model parameter to True')

if __name__ == "__main__":
    main()
