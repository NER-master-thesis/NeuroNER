import sys

import utils
import os
import pickle
import re
import time
import collections
import utils
import utils_nlp
import random
import copy
import numpy as np
from pprint import pprint


class Dataset(object):
    """A class for handling data sets."""

    def __init__(self, name='', verbose=False, debug=False):
        self.name = name
        self.verbose = verbose
        self.debug = debug
        self.embeddings_matrix = None

    def _parse_dataset(self, dataset_filepath, language):
        token_count = collections.defaultdict(lambda: 0)
        label_count = collections.defaultdict(lambda: 0)
        character_count = collections.defaultdict(lambda: 0)

        line_count = -1
        tokens = []
        labels = []
        new_token_sequence = []
        new_label_sequence = []
        if dataset_filepath:
            encoding = "UTF-8"
            if "combined" in dataset_filepath:
                label_index = 2
            else:
                label_index = -1
            with open(dataset_filepath, "r", encoding=encoding ) as file:
            #f = codecs.open(dataset_filepath, 'r', 'UTF-8')
                for line in file:
                    line_count += 1
                    line = line.strip().split(' ')
                    if len(line) == 0 or len(line[0]) == 0 or '-DOCSTART-' in line[0]:
                        if len(new_token_sequence) > 0:
                            labels.append(new_label_sequence)
                            tokens.append(new_token_sequence)
                            new_token_sequence = []
                            new_label_sequence = []
                        continue
                    if len(line) == 4  and line[-1] == "O":
                        continue
                    #if len(line) == 2:
                    #    tmp = [' ']
                    #    tmp.extend(line)
                    #    line = tmp
                    token = str(line[0])

                    try:
                        label = str(line[label_index])
                        if label == "NN":
                            a = 1
                    except IndexError:
                        print(line_count, line)
                        print(file.readline())
                        print(file.readline())
                        print(file.readline())
                        sys.exit()

                    token_count[token] += 1
                    label_count[label] += 1

                    new_token_sequence.append(token)
                    new_label_sequence.append(label)

                    for character in token:
                        character_count[character] += 1

                    # for debugging purposes
                    if self.debug and line_count > 100: break

                if len(new_token_sequence) > 0:
                    labels.append(new_label_sequence)
                    tokens.append(new_token_sequence)

        return labels, tokens, token_count, label_count, character_count

    @staticmethod
    def chunk(x, n):
        output = []
        size = int(len(x)/n)
        for i in range(0, len(x), size):
            if i < n-1:
                output.append(x[i:i + size])
            else:
                output.append(x[i:])
                return output

    def get_chunks(self, cv):
        #shuffle the args:
        shuffled_args = list(range(0,len(self.tokens['train'])))
        random.shuffle(shuffled_args)
        return self.chunk(shuffled_args, cv)

    def get_copy(self):
        return copy.deepcopy(self)

    def split(self, chunks, id_of_test):
        train_ids = [l for i in chunks[0:id_of_test] + chunks[id_of_test+1:] for l in i]
        test_ids = chunks[id_of_test]

        dataset_copy = self.get_copy()
        dataset_copy.token_indices['train'] = np.array(self.token_indices['train'])[train_ids]
        dataset_copy.tokens['train'] = np.array(self.tokens['train'])[train_ids]
        dataset_copy.characters['train'] = np.array(self.characters['train'])[train_ids]
        dataset_copy.character_indices['train'] = np.array(self.character_indices['train'])[train_ids]
        dataset_copy.token_lengths['train'] = np.array(self.token_lengths['train'])[train_ids]
        dataset_copy.sequence_lengths['train'] = np.array(self.sequence_lengths['train'])[train_ids]
        dataset_copy.longest_token_length_in_sequence['train'] = np.array(self.longest_token_length_in_sequence['train'])[train_ids]
        dataset_copy.label_indices['train'] = np.array(self.label_indices['train'])[train_ids]
        dataset_copy.labels['train'] = np.array(self.labels['train'])[train_ids]
        dataset_copy.label_vector_indices['train'] = np.array(self.label_vector_indices['train'])[train_ids]

        dataset_copy.token_indices['valid'] = np.array(self.token_indices['train'])[test_ids]
        dataset_copy.tokens['valid'] = np.array(self.tokens['train'])[test_ids]
        dataset_copy.characters['valid'] = np.array(self.characters['train'])[test_ids]
        dataset_copy.character_indices['valid'] = np.array(self.character_indices['train'])[test_ids]
        dataset_copy.token_lengths['valid'] = np.array(self.token_lengths['train'])[test_ids]
        dataset_copy.sequence_lengths['valid'] = np.array(self.sequence_lengths['train'])[test_ids]
        dataset_copy.longest_token_length_in_sequence['valid'] = \
        np.array(self.longest_token_length_in_sequence['train'])[test_ids]
        dataset_copy.label_indices['valid'] = np.array(self.label_indices['train'])[test_ids]
        dataset_copy.labels['valid'] = np.array(self.labels['train'])[test_ids]
        dataset_copy.label_vector_indices['valid'] = np.array(self.label_vector_indices['train'])[test_ids]

        return dataset_copy

    def load_vocab_word_embeddings(self, parameters):
        print('LOADING Vocab EMBEDDINGS')
        self.vocab_embeddings = []
        if parameters['use_pretrained_embeddings']:
            self.vocab_embeddings = utils_nlp.load_tokens_from_pretrained_token_embeddings(parameters)
        if self.verbose: print("len(embeddings_matrix): {0}".format(len(self.embeddings_matrix)))

    def load_embeddings_matrix(self, parameters):
        if parameters['use_pretrained_embeddings']:
            self.embeddings_matrix = utils.load_pickle(utils_nlp.get_embedding_file_path(parameters))

    def load_dataset(self, dataset_filepaths, parameters):
        '''
        dataset_filepaths : dictionary with keys 'train', 'valid', 'test'
        '''
        start_time = time.time()
        pprint('Load dataset... ')
        # Load pretraining dataset to ensure that index to label is compatible to the pretrained model,
        #   and that token embeddings that are learned in the pretrained model are loaded properly.
        all_tokens_in_pretraining_dataset = []
        if parameters['use_pretrained_model']:
            pretrained_model_folder = os.path.dirname(parameters['pretrained_model_checkpoint_filepath'])
            pretraining_dataset = pickle.load(open(os.path.join(pretrained_model_folder, 'dataset.pickle'), 'rb'))
            all_tokens_in_pretraining_dataset = pretraining_dataset.index_to_token.values()
            self.vocab_embeddings = all_tokens_in_pretraining_dataset


        remap_to_unk_count_threshold = 1
        self.PADDING_CHARACTER_INDEX = 1
        self.PADDING_TOKEN_INDEX = 1
        self.UNK_TOKEN_INDEX = 0
        self.UNK_CHARACTER_INDEX = 0
        self.tokens_mapped_to_unk = []
        self.UNK = '<UNK>'
        self.PAD = '<PAD>'
        self.unique_labels = []
        labels = {}
        tokens = {}
        characters = {}
        token_lengths = {}
        sequence_lengths = {}
        longest_token_length_in_sequence = {}
        label_count = {}
        token_count = {}
        character_count = {}


        for dataset_type in ['train', 'valid', 'test']:
            labels[dataset_type], tokens[dataset_type], token_count[dataset_type], label_count[dataset_type], character_count[dataset_type] \
                = self._parse_dataset(dataset_filepaths.get(dataset_type, None), parameters['language'])

            if self.verbose: print("dataset_type: {0}".format(dataset_type))
            if self.verbose: print("len(token_count[dataset_type]): {0}".format(len(token_count[dataset_type])))

        token_count['all'] = {}
        for token in list(token_count['train'].keys()) + list(token_count['valid'].keys()) + list(token_count['test'].keys()) :
            token_count['all'][token] = token_count['train'][token] + token_count['valid'][token] + token_count['test'][token]

        for dataset_type in dataset_filepaths.keys():
            if self.verbose: print("dataset_type: {0}".format(dataset_type))
            if self.verbose: print("len(token_count[dataset_type]): {0}".format(len(token_count[dataset_type])))

        character_count['all'] = {}
        for character in list(character_count['train'].keys()) + list(character_count['valid'].keys()) + list(character_count['test'].keys()):
            character_count['all'][character] = character_count['train'][character] + character_count['valid'][character] + character_count['test'][character]

        label_count['all'] = {}
        for character in list(label_count['train'].keys()) + list(label_count['valid'].keys()) + list(label_count['test'].keys()) :
            label_count['all'][character] = label_count['train'][character] + label_count['valid'][character] + label_count['test'][character]

        token_count['all'] = utils.order_dictionary(token_count['all'], 'value_key', reverse = True)
        label_count['all'] = utils.order_dictionary(label_count['all'], 'key', reverse = False)
        character_count['all'] = utils.order_dictionary(character_count['all'], 'value', reverse = True)
        if self.verbose: print('character_count[\'all\']: {0}'.format(character_count['all']))

        token_to_index = {}
        token_to_index[self.UNK] = self.UNK_TOKEN_INDEX
        token_to_index[self.PAD] = self.PADDING_TOKEN_INDEX
        iteration_number = 0
        number_of_unknown_tokens = 0
        if self.verbose: print("parameters['remap_unknown_tokens_to_unk']: {0}".format(parameters['remap_unknown_tokens_to_unk']))
        if self.verbose: print("len(token_count['train'].keys()): {0}".format(len(token_count['train'].keys())))
        for token, count in token_count['all'].items():
            if iteration_number == self.UNK_TOKEN_INDEX: iteration_number += 1
            if iteration_number == self.PADDING_TOKEN_INDEX: iteration_number += 1

            if parameters['remap_unknown_tokens_to_unk'] == 1 and \
                (token_count['train'][token] == 0 or \
                parameters['load_only_pretrained_token_embeddings']) and \
                not utils_nlp.is_token_in_pretrained_embeddings(token, self.vocab_embeddings, parameters) and \
                token not in all_tokens_in_pretraining_dataset:
                if self.verbose: print("token: {0}".format(token))
                if self.verbose: print("token.lower(): {0}".format(token.lower()))
                if self.verbose: print("re.sub('\d', '0', token.lower()): {0}".format(re.sub('\d', '0', token.lower())))
                if parameters['embedding_type'] == 'glove':
                    token_to_index[token] =  self.UNK_TOKEN_INDEX
                    number_of_unknown_tokens += 1
                    self.tokens_mapped_to_unk.append(token)
                elif parameters['embedding_type'] == 'fasttext':
                    token_to_index[token] = iteration_number
                    iteration_number += 1
                else:
                    raise AssertionError("Embedding type not recognized")
            else:
                token_to_index[token] = iteration_number
                iteration_number += 1
        if self.verbose: print("number_of_unknown_tokens: {0}".format(number_of_unknown_tokens))

        infrequent_token_indices = []
        for token, count in token_count['train'].items():
            if 0 < count <= remap_to_unk_count_threshold:
                infrequent_token_indices.append(token_to_index[token])
        if self.verbose: print("len(token_count['train']): {0}".format(len(token_count['train'])))
        if self.verbose: print("len(infrequent_token_indices): {0}".format(len(infrequent_token_indices)))

        label_count['all'] = utils.order_dictionary(label_count['all'], 'key', reverse = False)

        if parameters['use_pretrained_model']:
            self.unique_labels = sorted(list(pretraining_dataset.label_to_index.keys()))
            # Make sure labels are compatible with the pretraining dataset.
            for label in label_count['all']:
                if label not in pretraining_dataset.label_to_index:
                    raise AssertionError("The label {0} does not exist in the pretraining dataset. ".format(label) +
                                         "Please ensure that only the following labels exist in the dataset: {0}".format(', '.join(self.unique_labels)))
            label_to_index = pretraining_dataset.label_to_index.copy()
        else:
            label_to_index = {}
            iteration_number = 0
            for label, count in label_count['all'].items():
                label_to_index[label] = iteration_number
                iteration_number += 1
                self.unique_labels.append(label)
        self.PADDING_LABEL_INDEX = label_to_index['O']

        if self.verbose: print('self.unique_labels: {0}'.format(self.unique_labels))

        character_to_index = {}
        character_to_index[self.UNK] = self.UNK_CHARACTER_INDEX

        if parameters['use_pretrained_model']:
            # TODO: initialize character_to_index from saved pickle
            character_to_index = pretraining_dataset.character_to_index.copy()
        else:
            character_to_index[self.PAD] = self.PADDING_CHARACTER_INDEX
            iteration_number = 0
            for character, count in character_count['all'].items():
                if iteration_number == self.UNK_CHARACTER_INDEX: iteration_number += 1
                if iteration_number == self.PADDING_CHARACTER_INDEX: iteration_number += 1
                character_to_index[character] = iteration_number
                iteration_number += 1

        if self.verbose: print('token_count[\'train\'][0:10]: {0}'.format(list(token_count['train'].items())[0:10]))
        token_to_index = utils.order_dictionary(token_to_index, 'value', reverse = False)
        if self.verbose: print('token_to_index: {0}'.format(token_to_index))
        index_to_token = utils.reverse_dictionary(token_to_index)
        if parameters['remap_unknown_tokens_to_unk'] == 1: index_to_token[self.UNK_TOKEN_INDEX] = self.UNK
        if self.verbose: print('index_to_token: {0}'.format(index_to_token))

        if self.verbose: print('label_count[\'train\']: {0}'.format(label_count['train']))
        label_to_index = utils.order_dictionary(label_to_index, 'value', reverse = False)
        if self.verbose: print('label_to_index: {0}'.format(label_to_index))
        index_to_label = utils.reverse_dictionary(label_to_index)
        if self.verbose: print('index_to_label: {0}'.format(index_to_label))

        character_to_index = utils.order_dictionary(character_to_index, 'value', reverse = False)
        index_to_character = utils.reverse_dictionary(character_to_index)
        if self.verbose: print('character_to_index: {0}'.format(character_to_index))
        if self.verbose: print('index_to_character: {0}'.format(index_to_character))


        if self.verbose: print('labels[\'train\'][0:10]: {0}'.format(labels['train'][0:10]))
        if self.verbose: print('tokens[\'train\'][0:10]: {0}'.format(tokens['train'][0:10]))

        if self.verbose:
            # Print sequences of length 1 in train set
            for token_sequence, label_sequence in zip(tokens['train'], labels['train']):
                if len(label_sequence) == 1 and label_sequence[0] != 'O':
                    print("{0}\t{1}".format(token_sequence[0], label_sequence[0]))

        # Map tokens and labels to their indices
        token_indices = {}
        label_indices = {}
        character_indices = {}
        #character_indices_padded = {}
        for dataset_type in dataset_filepaths.keys():
            token_indices[dataset_type] = []
            characters[dataset_type] = []
            character_indices[dataset_type] = []
            token_lengths[dataset_type] = []
            sequence_lengths[dataset_type] = []
            longest_token_length_in_sequence[dataset_type] = []
            #character_indices_padded[dataset_type] = []
            for token_sequence in tokens[dataset_type]:
                token_indices[dataset_type].append([token_to_index[token] for token in token_sequence])
                characters[dataset_type].append([list(token) for token in token_sequence])
                character_indices[dataset_type].append([[character_to_index[character] for character in token] for token in token_sequence])
                token_lengths[dataset_type].append([len(token) for token in token_sequence])
                sequence_lengths[dataset_type].append(len(token_sequence))
                longest_token_length_in_sequence[dataset_type].append(max(token_lengths[dataset_type][-1]))

                #character_indices_padded[dataset_type].append([ utils.pad_list(temp_token_indices, longest_token_length_in_sequence, self.PADDING_CHARACTER_INDEX)
                #                                                for temp_token_indices in character_indices[dataset_type][-1]])

            label_indices[dataset_type] = []
            for label_sequence in labels[dataset_type]:
                label_indices[dataset_type].append([label_to_index[label] for label in label_sequence])

        if self.verbose: print('token_lengths[\'train\'][0][0:10]: {0}'.format(token_lengths['train'][0][0:10]))
        if self.verbose: print('characters[\'train\'][0][0:10]: {0}'.format(characters['train'][0][0:10]))
        if self.verbose: print('token_indices[\'train\'][0:10]: {0}'.format(token_indices['train'][0:10]))
        if self.verbose: print('label_indices[\'train\'][0:10]: {0}'.format(label_indices['train'][0:10]))
        if self.verbose: print('character_indices[\'train\'][0][0:10]: {0}'.format(character_indices['train'][0][0:10]))
        #if self.verbose: print('character_indices_padded[\'train\'][0][0:10]: {0}'.format(character_indices_padded['train'][0][0:10]))

        label_vector_indices = {}
        tmp_vector = [0] * len(self.unique_labels)
        tmp_vector[label_to_index["O"]] = 1
        self.PADDING_LABEL_VECTOR = tmp_vector
        for dataset_type in dataset_filepaths.keys():
            label_vector_indices[dataset_type] = []
            for label_indices_sequence in label_indices[dataset_type]:
                vector_sequence = []
                for indice in label_indices_sequence:
                    vector = [0] * len(self.unique_labels)
                    vector[indice] = 1
                    vector_sequence.append(vector)
                label_vector_indices[dataset_type].append(vector_sequence)

        if self.verbose: print('label_vector_indices[\'train\'][0:2]: {0}'.format(label_vector_indices['train'][0:2]))

        if self.verbose: print('len(label_vector_indices[\'train\']): {0}'.format(len(label_vector_indices['train'])))
        self.token_to_index = token_to_index
        self.index_to_token = index_to_token
        self.token_indices = token_indices
        self.label_indices = label_indices
        #self.character_indices_padded = character_indices_padded
        self.index_to_character = index_to_character
        self.character_to_index = character_to_index
        self.character_indices = character_indices
        self.token_lengths = token_lengths
        self.sequence_lengths = sequence_lengths
        self.longest_token_length_in_sequence = longest_token_length_in_sequence
        self.characters = characters
        self.tokens = tokens
        self.labels = labels
        self.label_vector_indices = label_vector_indices
        self.index_to_label = index_to_label
        self.label_to_index = label_to_index
        if self.verbose: print("len(self.token_to_index): {0}".format(len(self.token_to_index)))
        if self.verbose: print("len(self.index_to_token): {0}".format(len(self.index_to_token)))

        self.number_of_classes = len(self.unique_labels)
        self.vocabulary_size = len(self.index_to_token)
        self.alphabet_size = len(self.character_to_index)
        if self.verbose: print("self.number_of_classes: {0}".format(self.number_of_classes))
        if self.verbose: print("self.alphabet_size: {0}".format(self.alphabet_size))
        if self.verbose: print("self.vocabulary_size: {0}".format(self.vocabulary_size))

        # unique_labels_of_interest is used to compute F1-scores.
        self.unique_labels_of_interest = list(self.unique_labels)
        self.unique_labels_of_interest.remove('O')

        self.unique_label_indices_of_interest = []
        for lab in self.unique_labels_of_interest:
            self.unique_label_indices_of_interest.append(label_to_index[lab])

        self.infrequent_token_indices = infrequent_token_indices

        if self.verbose: print('self.unique_labels_of_interest: {0}'.format(self.unique_labels_of_interest))
        if self.verbose: print('self.unique_label_indices_of_interest: {0}'.format(self.unique_label_indices_of_interest))


        print(self.label_to_index)
        elapsed_time = time.time() - start_time
        print('done ({0:.2f} seconds)'.format(elapsed_time))

