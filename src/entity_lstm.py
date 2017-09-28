import subprocess
import tensorflow as tf
tf.set_random_seed(0)
import numpy as np
import codecs
import re
import time
import utils_tf
import utils_nlp
import os

def bidirectional_LSTM(input, hidden_state_dimension, initializer, sequence_length=None, output_sequence=True):

    with tf.variable_scope("bidirectional_LSTM"):
        if sequence_length == None:
            batch_size = 1
            sequence_length = tf.shape(input)[1]
            sequence_length = tf.expand_dims(sequence_length, axis=0, name='sequence_length')
        else:
            batch_size = tf.shape(sequence_length)[0]

        lstm_cell = {}
        initial_state = {}
        for direction in ["forward", "backward"]:
            with tf.variable_scope(direction):
                # LSTM cell
                lstm_cell[direction] = tf.contrib.rnn.CoupledInputForgetGateLSTMCell(hidden_state_dimension, forget_bias=1.0, initializer=initializer, state_is_tuple=True)
                # initial state: http://stackoverflow.com/questions/38441589/tensorflow-rnn-initial-state
                initial_cell_state = tf.get_variable("initial_cell_state", shape=[1, hidden_state_dimension], dtype=tf.float32, initializer=initializer)
                initial_output_state = tf.get_variable("initial_output_state", shape=[1, hidden_state_dimension], dtype=tf.float32, initializer=initializer)
                c_states = tf.tile(initial_cell_state, tf.stack([batch_size, 1]))
                h_states = tf.tile(initial_output_state, tf.stack([batch_size, 1]))
                initial_state[direction] = tf.contrib.rnn.LSTMStateTuple(c_states, h_states)

        # sequence_length must be provided for tf.nn.bidirectional_dynamic_rnn due to internal bug
        outputs, final_states = tf.nn.bidirectional_dynamic_rnn(lstm_cell["forward"],
                                                                    lstm_cell["backward"],
                                                                    input,
                                                                    dtype=tf.float32,
                                                                    sequence_length=sequence_length,
                                                                    initial_state_fw=initial_state["forward"],
                                                                    initial_state_bw=initial_state["backward"])
        if output_sequence == True:
            outputs_forward, outputs_backward = outputs
            output = tf.concat([outputs_forward, outputs_backward], axis=-1, name='output_sequence')
        else:
            # max pooling
#             outputs_forward, outputs_backward = outputs
#             output = tf.concat([outputs_forward, outputs_backward], axis=2, name='output_sequence')
#             output = tf.reduce_max(output, axis=1, name='output')
            # last pooling
            final_states_forward, final_states_backward = final_states
            output = tf.concat([final_states_forward[1], final_states_backward[1]], axis=-1, name='output')

    return output

def bidirectional_GRU(input, hidden_state_dimension, hidden_layers, initializer, sequence_length=None, output_sequence=True):
    print("GRU neuron")
    with tf.variable_scope("bidirectional_gru"):

        #num_sentence_by_batch = tf.shape(sequence_length)[0]
        #num_token_by_sentence = tf.shape(sequence_length)[1]

        gru_cell = {}
        initial_state = {}
        for direction in ["forward", "backward"]:
            with tf.variable_scope(direction):
                # GRU cell
                gru_cell[direction] = tf.contrib.rnn.GRUCell(hidden_state_dimension) # [tf.contrib.rnn.GRUCell(hidden_state_dimension) for _ in range(hidden_layers)]
                # initial state: http://stackoverflow.com/questions/38441589/tensorflow-rnn-initial-state
                #initial_output_state = tf.get_variable("initial_output_state", shape=[1, hidden_state_dimension],
                  #                                    dtype=tf.float32, initializer=initializer)
                #initial_output_state = tf.tile(initial_output_state, tf.stack([num_token_by_sentence,1]))
                ##initial_state[direction] = tf.tile(initial_output_state, tf.stack([num_sentence_by_batch,num_token_by_sentence]))

        # sequence_length must be provided for tf.nn.bidirectional_dynamic_rnn due to internal bug
        outputs, final_states = tf.nn.bidirectional_dynamic_rnn(gru_cell["forward"],
                                                                gru_cell["backward"],
                                                                input,
                                                                dtype=tf.float32,
                                                                sequence_length=sequence_length)
        #                                                        initial_state_fw=initial_state["forward"],
        #                                                        initial_state_bw=initial_state["backward"])
        if output_sequence == True:
            outputs_forward, outputs_backward = outputs
            output = tf.concat([outputs_forward, outputs_backward], axis=-1, name='output_sequence')
        else:
            # max pooling
            #             outputs_forward, outputs_backward = outputs
            #             output = tf.concat([outputs_forward, outputs_backward], axis=2, name='output_sequence')
            #             output = tf.reduce_max(output, axis=1, name='output')
            # last pooling
            final_states_forward, final_states_backward = final_states
            output = tf.concat([final_states_forward, final_states_backward], axis=-1, name='output')

    return output



class EntityLSTM(object):
    """
    An LSTM architecture for named entity recognition.
    Uses a character embedding layer followed by an LSTM to generate vector representation from characters for each token.
    Then the character vector is concatenated with token embedding vector, which is input to another LSTM  followed by a CRF layer.
    """
    def __init__(self, dataset, parameters):

        self.verbose = False

        # Placeholders for input, output and dropout
        self.input_token_indices = tf.placeholder(tf.int32, [None,None], name="input_token_indices") #[batch, sequence_length]
        self.input_sequence_lengths = tf.placeholder(tf.int32, [None], name="input_sequence_lengths") #[batch_size]
        self.input_label_indices_vector = tf.placeholder(tf.int32, [None,None, dataset.number_of_classes], name="input_label_indices_vector")
        self.input_label_indices_flat = tf.placeholder(tf.int32, [None,None], name="input_label_indices_flat") #[batch_size, max_sentence_length]
        self.input_token_character_indices = tf.placeholder(tf.int32, [None,None, None], name="input_token_character_indices")# [batch, sequence_length, token_length]
        self.input_token_lengths = tf.placeholder(tf.int32, [None,None], name="input_token_lengths") # [batch, sequence_length]
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

        self.embedding_dim = parameters['embedding_dimension']
        batch_size = tf.shape(self.input_token_character_indices)[0]
        sentence_size = tf.shape(self.input_token_character_indices)[1]
        token_size = tf.shape(self.input_token_character_indices)[2]
        # Internal parameters
        initializer = tf.contrib.layers.xavier_initializer()

        if parameters['use_character_lstm']:
            # Character-level LSTM
            # Idea: reshape so that we have a tensor [number_of_token, max_token_length, token_embeddings_size], which we pass to the LSTM

            # Character embedding layer
            with tf.variable_scope("character_embedding"):
                self.character_embedding_weights = tf.get_variable(
                    "character_embedding_weights",
                    shape=[dataset.alphabet_size, parameters['character_embedding_dimension']],
                    initializer=initializer)
                embedded_characters = tf.nn.embedding_lookup(self.character_embedding_weights, self.input_token_character_indices, name='embedded_characters')
                if self.verbose: print("embedded_characters: {0}".format(embedded_characters))
                utils_tf.variable_summaries(self.character_embedding_weights)

            # Character LSTM layer
            with tf.variable_scope('character_lstm') as vs:
                #batch_size = tf.shape(embedded_characters)[0]
                #sentence_size = tf.shape(embedded_characters)[1]
                #token_size = tf.shape(embedded_characters)[2]
                embedded_characters = tf.reshape(embedded_characters,
                                                 [batch_size*sentence_size, token_size, parameters['character_embedding_dimension']])
                input_token_lengths = tf.reshape(self.input_token_lengths, [-1])
                if parameters['gru_neuron']:
                    character_lstm_output = bidirectional_GRU(embedded_characters,
                                                               parameters['character_lstm_hidden_state_dimension'],
                                                               parameters['character_hidden_layer'],
                                                               initializer,
                                                               sequence_length=input_token_lengths,
                                                               output_sequence=False)

                else:
                    character_lstm_output = bidirectional_LSTM(embedded_characters, parameters['character_lstm_hidden_state_dimension'], initializer,
                                                           sequence_length=input_token_lengths, output_sequence=False)
                self.character_lstm_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=vs.name)

                character_lstm_output = tf.reshape(character_lstm_output, [batch_size, sentence_size, 2*parameters['character_lstm_hidden_state_dimension']])

        # Token embedding layer
        with tf.variable_scope("token_embedding"):
            self.token_embedding_weights = tf.get_variable(
                "token_embedding_weights",
                shape=[dataset.vocabulary_size, parameters['embedding_dimension']],
                initializer=initializer,
                trainable=not parameters['freeze_token_embeddings'])
            embedded_tokens = tf.nn.embedding_lookup(self.token_embedding_weights, self.input_token_indices)
            utils_tf.variable_summaries(self.token_embedding_weights)

        # Concatenate character LSTM outputs and token embeddings
        if parameters['use_character_lstm']:
            with tf.variable_scope("concatenate_token_and_character_vectors"):
                if self.verbose: print('embedded_tokens: {0}'.format(embedded_tokens))
                token_lstm_input = tf.concat([character_lstm_output, embedded_tokens], axis=-1, name='token_lstm_input')
                if self.verbose: print("token_lstm_input: {0}".format(token_lstm_input))
        else:
            token_lstm_input = embedded_tokens

        # Add dropout
        with tf.variable_scope("dropout"):
            token_lstm_input_drop = tf.nn.dropout(token_lstm_input, self.dropout_keep_prob, name='token_lstm_input_drop')
            if self.verbose: print("token_lstm_input_drop: {0}".format(token_lstm_input_drop))
            # https://www.tensorflow.org/api_guides/python/contrib.rnn
            # Prepare data shape to match `rnn` function requirements
            # Current data input shape: (batch_size, n_steps, n_input)
            # Required shape: 'n_steps' tensors list of shape (batch_size, n_input)
     #       token_lstm_input_drop_expanded = tf.expand_dims(token_lstm_input_drop, axis=0, name='token_lstm_input_drop_expanded')
            token_lstm_input_drop_expanded = token_lstm_input_drop
            if self.verbose: print("token_lstm_input_drop_expanded: {0}".format(token_lstm_input_drop_expanded))

        # Token LSTM layer
        with tf.variable_scope('token_lstm') as vs:
            if parameters['gru_neuron']:
                token_lstm_output = bidirectional_GRU(token_lstm_input_drop_expanded,
                                                       parameters['token_lstm_hidden_state_dimension'],
                                                       parameters['token_hidden_layer'],
                                                       initializer=initializer,
                                                       sequence_length = self.input_sequence_lengths,
                                                       output_sequence=True)
            else:
                token_lstm_output = bidirectional_LSTM(token_lstm_input_drop_expanded,
                                                       parameters['token_lstm_hidden_state_dimension'],
                                                       initializer=initializer,
                                                       sequence_length = self.input_sequence_lengths,
                                                       output_sequence=True)
            self.token_lstm_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=vs.name)

        # Needed only if Bidirectional LSTM is used for token level
        with tf.variable_scope("feedforward_after_lstm") as vs:
            token_lstm_output_squeezed = tf.reshape(token_lstm_output, [batch_size * sentence_size,2*parameters['token_lstm_hidden_state_dimension']])

            W = tf.get_variable(
                "W",
                shape=[2 * parameters['token_lstm_hidden_state_dimension'], parameters['token_lstm_hidden_state_dimension']],
                initializer=initializer)
            b = tf.Variable(tf.constant(0.0, shape=[parameters['token_lstm_hidden_state_dimension']]), name="bias")
            outputs = tf.nn.xw_plus_b(token_lstm_output_squeezed, W, b, name="output_before_tanh")
            outputs = tf.nn.tanh(outputs, name="output_after_tanh")
            utils_tf.variable_summaries(W)
            utils_tf.variable_summaries(b)
            self.token_lstm_variables += tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=vs.name)

        with tf.variable_scope("feedforward_before_crf") as vs:
            W = tf.get_variable(
                "W",
                shape=[parameters['token_lstm_hidden_state_dimension'], dataset.number_of_classes],
                initializer=initializer)
            b = tf.Variable(tf.constant(0.0, shape=[dataset.number_of_classes]), name="bias")
            scores = tf.nn.xw_plus_b(outputs, W, b, name="scores")
            self.unary_scores = tf.reshape(scores, [batch_size, sentence_size, dataset.number_of_classes])
            self.predictions = tf.argmax(self.unary_scores, 1, name="predictions")
            utils_tf.variable_summaries(W)
            utils_tf.variable_summaries(b)
            self.feedforward_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=vs.name)

        # Add dropout
        #with tf.variable_scope("dropout"):
        #    self.unary_scores = tf.nn.dropout(self.unary_scores, self.dropout_keep_prob,
        #                                          name='crf__input_drop')
        #    if self.verbose: print("crf_input_drop: {0}".format(self.unary_scores))
            # https://www.tensorflow.org/api_guides/python/contrib.rnn
            # Prepare data shape to match `rnn` function requirements
            # Current data input shape: (batch_size, n_steps, n_input)
            # Required shape: 'n_steps' tensors list of shape (batch_size, n_input)
            #       token_lstm_input_drop_expanded = tf.expand_dims(token_lstm_input_drop, axis=0, name='token_lstm_input_drop_expanded')
        # CRF layer
        if parameters['use_crf']:
            with tf.variable_scope("crf") as vs:
                # Add start and end tokens
                #small_score = -1000.0
                #large_score = 0.0
                #sequence_length = tf.shape(self.unary_scores)[1]
                #tmp = tf.tile( tf.constant(small_score, shape=[1, 2]) , [sentence_size, 1])
                #unary_scores_with_start_and_end = tf.concat([self.unary_scores, tmp], 1)
                #start_unary_scores = [[small_score] * dataset.number_of_classes + [large_score, small_score]]
                #end_unary_scores = [[small_score] * dataset.number_of_classes + [small_score, large_score]]
                #self.unary_scores = tf.concat([start_unary_scores, unary_scores_with_start_and_end, end_unary_scores], 0)
                #start_index = dataset.number_of_classes
                #end_index = dataset.number_of_classes + 1
                #input_label_indices_flat_with_start_and_end = tf.concat([ tf.constant(start_index, shape=[1]), self.input_label_indices_flat, tf.constant(end_index, shape=[1]) ], 0)

                # Apply CRF layer
                #sequence_length = tf.shape(self.unary_scores)[0]
                #sequence_lengths = tf.expand_dims(sequence_length, axis=0, name='sequence_lengths')
                #unary_scores_expanded = tf.expand_dims(self.unary_scores, axis=0, name='unary_scores_expanded')
                #input_label_indices_flat_batch = tf.expand_dims(input_label_indices_flat_with_start_and_end, axis=0, name='input_label_indices_flat_batch')
                #if self.verbose: print('unary_scores_expanded: {0}'.format(unary_scores_expanded))
                #if self.verbose: print('input_label_indices_flat_batch: {0}'.format(input_label_indices_flat_batch))
                #if self.verbose: print("sequence_lengths: {0}".format(sequence_lengths))
                # https://github.com/tensorflow/tensorflow/tree/master/tensorflow/contrib/crf
                # Compute the log-likelihood of the gold sequences and keep the transition params for inference at test time.

                self.transition_parameters=tf.get_variable(
                    "transitions",
                    #shape=[dataset.number_of_classes+2, dataset.number_of_classes+2],
                    shape=[dataset.number_of_classes, dataset.number_of_classes],
                    initializer=initializer)
                utils_tf.variable_summaries(self.transition_parameters)
                #log_likelihood, _ = tf.contrib.crf.crf_log_likelihood(
                #    unary_scores_expanded, input_label_indices_flat_batch, sequence_lengths, transition_params=self.transition_parameters)

                log_likelihood, _ = tf.contrib.crf.crf_log_likelihood(
                         self.unary_scores, self.input_label_indices_flat, self.input_sequence_lengths, transition_params=self.transition_parameters)

                #regularizer = tf.nn.l2_loss(W)
                self.loss =  tf.reduce_mean(-log_likelihood, name='cross_entropy_mean_loss')
                self.accuracy = tf.constant(1)

                self.crf_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=vs.name)

        # Do not use CRF layer
        else:
            #with tf.variable_scope("crf") as vs:
            #    self.transition_parameters = tf.get_variable(
            #        "transitions",
            #        shape=[dataset.number_of_classes+2, dataset.number_of_classes+2],
            #        initializer=initializer)
            #    utils_tf.variable_summaries(self.transition_parameters)
            #    self.crf_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=vs.name)

            # Calculate mean cross-entropy loss
            with tf.variable_scope("loss"):
                self.unary_scores = tf.reshape(self.unary_scores, [-1,dataset.number_of_classes])
                self.input_label_indices_vector = tf.reshape(self.input_label_indices_vector, [-1,dataset.number_of_classes])
                losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.unary_scores, labels=self.input_label_indices_vector, name='softmax')
                mask = tf.sequence_mask(self.input_sequence_lengths)
                losses = tf.boolean_mask(losses, mask)
                self.loss = tf.reduce_mean(losses, name='cross_entropy_mean_loss')
            with tf.variable_scope("accuracy"):
                correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_label_indices_vector, 1))
                self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, 'float'), name='accuracy')
                self. unary_scores = tf.reshape(self.unary_scores, [batch_size,-1, dataset.number_of_classes])
                self.input_label_indices_vector = tf.reshape(self.input_label_indices_vector,
                                                             [batch_size, -1, dataset.number_of_classes])
        self.define_training_procedure(parameters)
        self.summary_op = tf.summary.merge_all()

    def define_training_procedure(self, parameters):
        # Define training procedure
        self.global_step = tf.Variable(0, name="global_step", trainable=False)
        if parameters['optimizer'] == 'adam':
            self.optimizer = tf.train.AdamOptimizer(parameters['learning_rate'])
        elif parameters['optimizer'] == 'sgd':
            self.optimizer = tf.train.GradientDescentOptimizer(parameters['learning_rate'])
        elif parameters['optimizer'] == 'adadelta':
            self.optimizer = tf.train.AdadeltaOptimizer(parameters['learning_rate'])
        else:
            raise ValueError('The lr_method parameter must be either adadelta, adam or sgd.')

        grads_and_vars = self.optimizer.compute_gradients(self.loss)
        if parameters['gradient_clipping_value']:
            grads_and_vars = [(tf.clip_by_value(grad, -parameters['gradient_clipping_value'], parameters['gradient_clipping_value']), var) 
                              for grad, var in grads_and_vars]
        # By defining a global_step variable and passing it to the optimizer we allow TensorFlow handle the counting of training steps for us.
        # The global step will be automatically incremented by one every time you execute train_op.
        self.train_op = self.optimizer.apply_gradients(grads_and_vars, global_step=self.global_step)

    # TODO: maybe move out of the class?
    def load_pretrained_token_embeddings(self, sess, dataset, parameters):
        if not parameters['use_pretrained_embeddings']:
            return
        # Load embeddings
        start_time = time.time()
        print('Load token embeddings... ')

        initial_weights = sess.run(self.token_embedding_weights.read_value())
        number_of_loaded_word_vectors = 0
        number_of_token_original_case_found = 0
        number_of_token_lowercase_found = 0
        number_of_token_digits_replaced_with_zeros_found = 0
        number_of_token_lowercase_and_digits_replaced_with_zeros_found = 0
        number_of_token_not_found = 0
        tokens_list = []
        original_token_list = []
        token_not_in_embedding = []
        if "fasttext" in parameters['embedding_type']:
            for token in dataset.token_to_index.keys():
                if token in dataset.vocab_embeddings:
                    #initial_weights[dataset.token_to_index[token]] = dataset.embeddings_matrix[token]
                    tokens_list.append(token)
                    original_token_list.append(token)
                    number_of_token_original_case_found += 1
                elif parameters['check_for_lowercase'] and token.lower() in dataset.vocab_embeddings:
                    #initial_weights[dataset.token_to_index[token]] = dataset.embeddings_matrix[token.lower()]
                    tokens_list.append(token.lower())
                    original_token_list.append(token)
                    number_of_token_lowercase_found += 1
                elif parameters['check_for_digits_replaced_with_zeros'] and re.sub('\d', '0', token) in dataset.vocab_embeddings:
                    #initial_weights[dataset.token_to_index[token]] = dataset.embeddings_matrix[re.sub('\d', '0', token)]
                    tokens_list.append(re.sub('\d', '0', token))
                    original_token_list.append(token)
                    number_of_token_digits_replaced_with_zeros_found += 1
                elif parameters['check_for_lowercase'] and parameters['check_for_digits_replaced_with_zeros'] and re.sub('\d', '0', token.lower()) in dataset.vocab_embeddings:
                    #initial_weights[dataset.token_to_index[token]] = dataset.embeddings_matrix[re.sub('\d', '0', token.lower())]
                    tokens_list.append(re.sub('\d', '0', token.lower()))
                    original_token_list.append(token)
                    number_of_token_lowercase_and_digits_replaced_with_zeros_found += 1
                else:
                    if parameters['embedding_type'] == 'fasttext':
                        tokens_list.append(token)
                    else:
                        token_not_in_embedding.append(token)
                        number_of_token_not_found += 1
                    continue #TODO REMOVE
                    #initial_weights[dataset.token_to_index[token]] = dataset.embeddings_matrix[token]
                number_of_loaded_word_vectors += 1
            with open("tmp.txt", "w") as file:
                file.write(" ".join(tokens_list))
            fasttext_script = os.path.join("..", "fastText", "fasttext")
            shell_command = '{0} print-word-vectors {1} < {2}'.format(fasttext_script, utils_nlp.get_embedding_file_path_fasttext(parameters), "tmp.txt")
            output = subprocess.check_output(shell_command, shell=True)
            list_embeddings = output.decode().split("\n")
            #for token, emb in zip(dataset.token_to_index.keys(), list_embeddings):
            for token, emb in zip(original_token_list, list_embeddings):
                idx = dataset.token_to_index[token]
                initial_weights[idx] = list(map(float, emb.split()[-int(parameters["embedding_dimension"]):]))

            for token in token_not_in_embedding:
                random_emb = np.random.random(self.embedding_dim)
                random_emb /= np.linalg.norm(random_emb)
                idx = dataset.token_to_index[token]
                initial_weights[idx] = random_emb
        elif parameters['embedding_type'] in ["glove", "polyglot"]:
            dataset.load_embeddings_matrix(parameters)
            for token in dataset.token_to_index.keys():
                if token in dataset.embeddings_matrix:
                    initial_weights[dataset.token_to_index[token]] = dataset.embeddings_matrix.get(token)
                    number_of_token_original_case_found += 1
                elif parameters['check_for_lowercase'] and token.lower() in dataset.embeddings_matrix:
                    initial_weights[dataset.token_to_index[token]] = dataset.embeddings_matrix.get(token.lower())
                    number_of_token_lowercase_found += 1
                elif parameters['check_for_digits_replaced_with_zeros'] and re.sub('\d', '0', token) in dataset.embeddings_matrix:
                    initial_weights[dataset.token_to_index[token]] = dataset.embeddings_matrix.get(re.sub('\d', '0', token))
                    number_of_token_digits_replaced_with_zeros_found += 1
                elif parameters['check_for_lowercase'] and parameters['check_for_digits_replaced_with_zeros'] and re.sub('\d', '0', token.lower()) in dataset.embeddings_matrix:
                    initial_weights[dataset.token_to_index[token]] = dataset.embeddings_matrix.get(re.sub('\d', '0', token.lower()))
                    number_of_token_lowercase_and_digits_replaced_with_zeros_found += 1
                else:
                    number_of_token_not_found += 1
                    random_emb = np.random.random(self.embedding_dim)
                    random_emb /= np.linalg.norm(random_emb)
                    initial_weights[dataset.token_to_index[token]] = random_emb

                number_of_loaded_word_vectors += 1
        else:
            raise AssertionError("Embedding type not recognized")

        elapsed_time = time.time() - start_time
        print('done ({0:.2f} seconds)'.format(elapsed_time))
        print("number_of_token_original_case_found: {0}".format(number_of_token_original_case_found))
        print("number_of_token_lowercase_found: {0}".format(number_of_token_lowercase_found))
        print("number_of_token_digits_replaced_with_zeros_found: {0}".format(number_of_token_digits_replaced_with_zeros_found))
        print("number_of_token_lowercase_and_digits_replaced_with_zeros_found: {0}".format(number_of_token_lowercase_and_digits_replaced_with_zeros_found))
        print('number_of_loaded_word_vectors: {0}'.format(number_of_loaded_word_vectors))
        print('number_of_token_not_found: {0}'.format(number_of_token_not_found))
        print("dataset.vocabulary_size: {0}".format(dataset.vocabulary_size))
        sess.run(self.token_embedding_weights.assign(initial_weights))


    def load_embeddings_from_pretrained_model(self, sess, dataset, pretraining_dataset, pretrained_embedding_weights, embedding_type='token'):
        if embedding_type == 'token':
            embedding_weights = self.token_embedding_weights
            index_to_string = dataset.index_to_token
            pretraining_string_to_index = pretraining_dataset.token_to_index
        elif embedding_type == 'character':
            embedding_weights = self.character_embedding_weights
            index_to_string = dataset.index_to_character
            pretraining_string_to_index = pretraining_dataset.character_to_index
        # Load embeddings
        start_time = time.time()
        print('Load {0} embeddings from pretrained model... '.format(embedding_type), end='', flush=True)
        initial_weights = sess.run(embedding_weights.read_value())

        if embedding_type == 'token':
            initial_weights[dataset.UNK_TOKEN_INDEX] = pretrained_embedding_weights[pretraining_dataset.UNK_TOKEN_INDEX]
        elif embedding_type == 'character':
            initial_weights[dataset.PADDING_CHARACTER_INDEX] = pretrained_embedding_weights[pretraining_dataset.PADDING_CHARACTER_INDEX]

        number_of_loaded_vectors = 1
        for index, string in index_to_string.items():
            if index == dataset.UNK_TOKEN_INDEX:
                continue
            if string in pretraining_string_to_index.keys():
                initial_weights[index] = pretrained_embedding_weights[pretraining_string_to_index[string]]
                number_of_loaded_vectors += 1
        elapsed_time = time.time() - start_time
        print('done ({0:.2f} seconds)'.format(elapsed_time))
        print("number_of_loaded_vectors: {0}".format(number_of_loaded_vectors))
        if embedding_type == 'token':
            print("dataset.vocabulary_size: {0}".format(dataset.vocabulary_size))
        elif embedding_type == 'character':
            print("dataset.alphabet_size: {0}".format(dataset.alphabet_size))
        sess.run(embedding_weights.assign(initial_weights))

