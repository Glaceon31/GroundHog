import numpy
import logging
import pprint
import operator
import itertools

import theano
import theano.tensor as TT
from theano.ifelse import ifelse
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

from groundhog.layers import\
        Layer,\
        MultiLayer,\
        SoftmaxLayer,\
        HierarchicalSoftmaxLayer,\
        LSTMLayer, \
        RecurrentLayer,\
        RecursiveConvolutionalLayer,\
        UnaryOp,\
        Shift,\
        LastState,\
        DropOp,\
        Concatenate
from groundhog.models import LM_Model
from groundhog.datasets import PytablesBitextIterator
from groundhog.utils import sample_zeros, sample_weights_orth, init_bias, sample_weights_classic
from encdec import\
        create_padded_batch,\
        get_batch_iterator,\
        RecurrentLayerWithSearch,\
        ReplicateLayer,\
        PadLayer,\
        ZeroLayer,\
        none_if_zero,\
        Maxout,\
        parse_input,\
        prefix_lookup
import groundhog.utils as utils
from NTMLayer import NTMLayer, NTMLayerWithSearch

logger = logging.getLogger(__name__)

class NTMEncoderDecoderBase(object):

    def _create_only_embedding_layer(self):
        logger.debug("_create_only_embedding_layer")
        self.approx_embedder = MultiLayer(
            self.rng,
            n_in=self.state['n_sym_source']
                if self.prefix.find("enc") >= 0
                else self.state['n_sym_target'],
            n_hids=[self.state['rank_n_approx']],
            activation=[self.state['rank_n_activ']],
            name='{}_approx_embdr'.format(self.prefix),
            **self.default_kwargs)

    def _create_embedding_layers(self):
        logger.debug("_create_embedding_layers")
        self.approx_embedder = MultiLayer(
            self.rng,
            n_in=self.state['n_sym_source']
                if self.prefix.find("enc") >= 0
                else self.state['n_sym_target'],
            n_hids=[self.state['rank_n_approx']],
            activation=[self.state['rank_n_activ']],
            name='{}_approx_embdr'.format(self.prefix),
            **self.default_kwargs)

        # We have 3 embeddings for each word in each level,
        # the one used as input,
        # the one used to control resetting gate,
        # the one used to control update gate.
        self.input_embedders = [lambda x : 0] * self.num_levels
        self.reset_embedders = [lambda x : 0] * self.num_levels
        self.update_embedders = [lambda x : 0] * self.num_levels
        embedder_kwargs = dict(self.default_kwargs)
        embedder_kwargs.update(dict(
            n_in=self.state['rank_n_approx'],
            n_hids=[self.state['dim'] * self.state['dim_mult']],
            activation=['lambda x:x']))
        for level in range(self.num_levels):
            self.input_embedders[level] = MultiLayer(
                self.rng,
                name='{}_input_embdr_{}'.format(self.prefix, level),
                **embedder_kwargs)
            if prefix_lookup(self.state, self.prefix, 'rec_gating'):
                self.update_embedders[level] = MultiLayer(
                    self.rng,
                    learn_bias=False,
                    name='{}_update_embdr_{}'.format(self.prefix, level),
                    **embedder_kwargs)
            if prefix_lookup(self.state, self.prefix, 'rec_reseting'):
                self.reset_embedders[level] =  MultiLayer(
                    self.rng,
                    learn_bias=False,
                    name='{}_reset_embdr_{}'.format(self.prefix, level),
                    **embedder_kwargs)

    def _create_inter_level_layers(self):
        logger.debug("_create_inter_level_layers")
        inter_level_kwargs = dict(self.default_kwargs)
        inter_level_kwargs.update(
                n_in=self.state['dim'],
                n_hids=self.state['dim'] * self.state['dim_mult'],
                activation=['lambda x:x'])

        self.inputers = [0] * self.num_levels
        self.reseters = [0] * self.num_levels
        self.updaters = [0] * self.num_levels
        for level in range(1, self.num_levels):
            self.inputers[level] = MultiLayer(self.rng,
                    name="{}_inputer_{}".format(self.prefix, level),
                    **inter_level_kwargs)
            if prefix_lookup(self.state, self.prefix, 'rec_reseting'):
                self.reseters[level] = MultiLayer(self.rng,
                    name="{}_reseter_{}".format(self.prefix, level),
                    **inter_level_kwargs)
            if prefix_lookup(self.state, self.prefix, 'rec_gating'):
                self.updaters[level] = MultiLayer(self.rng,
                    name="{}_updater_{}".format(self.prefix, level),
                    **inter_level_kwargs)

    def _create_transition_layers(self, init_memory_param=True, head_fn = None):
        logger.debug("_create_transition_layers")
        self.transitions = []
        if not head_fn:
            head_fn = self.state['head_fn'] 
        rec_layer = eval(prefix_lookup(self.state, self.prefix, 'rec_layer'))
        print rec_layer
        add_args = dict()
        if rec_layer == NTMLayerWithSearch or rec_layer == NTMLayer:
            print 1
            add_args = dict(rank_n_approx=self.state['rank_n_approx'],
                    memory_weight = self.state['memory_weight'],
                    memory_size=self.state['memory_size'],
                    memory_dim=self.state['memory_dim'],
                    head_num=self.state['head_num'],
                    init_memory_weight=init_memory_param)
        if rec_layer == NTMLayerWithSearch or rec_layer == RecurrentLayerWithSearch:
            #add_args = dict(c_dim=self.state['c_dim'])
            add_args['c_dim'] = self.state['c_dim']
        print add_args
        for level in range(self.num_levels):
            self.transitions.append(rec_layer(
                    self.rng,
                    n_hids=self.state['dim'],
                    activation=prefix_lookup(self.state, self.prefix, 'activ'),
                    bias_scale=self.state['bias'],
                    init_fn=(self.state['rec_weight_init_fn']
                        if not self.skip_init
                        else "sample_zeros"),
                    scale=prefix_lookup(self.state, self.prefix, 'rec_weight_scale'),
                    
                    weight_noise=self.state['weight_noise_rec'],
                    head_fn=head_fn,
                    dropout=self.state['dropout_rec'],
                    gating=prefix_lookup(self.state, self.prefix, 'rec_gating'),
                    gater_activation=prefix_lookup(self.state, self.prefix, 'rec_gater'),
                    reseting=prefix_lookup(self.state, self.prefix, 'rec_reseting'),
                    reseter_activation=prefix_lookup(self.state, self.prefix, 'rec_reseter'),
                    name='{}_transition_{}'.format(self.prefix, level),
                    **add_args))

class NTMEncoder(NTMEncoderDecoderBase):

    def __init__(self, state, rng, prefix='enc', skip_init=False):
        self.state = state
        self.rng = rng
        self.prefix = prefix
        self.skip_init = skip_init

        self.num_levels = self.state['encoder_stack']

        # support multiple gating/memory units
        if 'dim_mult' not in self.state:
            self.state['dim_mult'] = 1.
        if 'hid_mult' not in self.state:
            self.state['hid_mult'] = 1.

    def create_layers(self):
        """ Create all elements of Encoder's computation graph"""

        self.default_kwargs = dict(
            init_fn=self.state['weight_init_fn'] if not self.skip_init else "sample_zeros",
            weight_noise=self.state['weight_noise'],
            scale=self.state['weight_scale'])

        self._create_embedding_layers()
        self._create_transition_layers(init_memory_param = self.state['encoder_memory_param'],head_fn = self.state['encoder_head_fn'])
        self._create_inter_level_layers()
        self._create_representation_layers()

    def _create_representation_layers(self):
        logger.debug("_create_representation_layers")
        # If we have a stack of RNN, then their last hidden states
        # are combined with a maxout layer.
        self.repr_contributors = [None] * self.num_levels
        for level in range(self.num_levels):
            self.repr_contributors[level] = MultiLayer(
                self.rng,
                n_in=self.state['dim'],
                n_hids=[self.state['dim'] * self.state['maxout_part']],
                activation=['lambda x: x'],
                name="{}_repr_contrib_{}".format(self.prefix, level),
                **self.default_kwargs)
        self.repr_calculator = UnaryOp(
                activation=eval(self.state['unary_activ']),
                name="{}_repr_calc".format(self.prefix))

    def build_encoder(self, x,
            x_mask=None,
            use_noise=False,
            approx_embeddings=None,
            return_hidden_layers=False):
        """Create the computational graph of the RNN Encoder

        :param x:
            input variable, either vector of word indices or
            matrix of word indices, where each column is a sentence

        :param x_mask:
            when x is a matrix and input sequences are
            of variable length, this 1/0 matrix is used to specify
            the matrix positions where the input actually is

        :param use_noise:
            turns on addition of noise to weights
            (UNTESTED)

        :param approx_embeddings:
            forces encoder to use given embeddings instead of its own

        :param return_hidden_layers:
            if True, encoder returns all the activations of the hidden layer
            (WORKS ONLY IN NON-HIERARCHICAL CASE)
        """

        # Low rank embeddings of all the input words.
        # Shape in case of matrix input:
        #   (max_seq_len * batch_size, rank_n_approx),
        #   where max_seq_len is the maximum length of batch sequences.
        # Here and later n_words = max_seq_len * batch_size.
        # Shape in case of vector input:
        #   (seq_len, rank_n_approx)
        if not approx_embeddings:
            approx_embeddings = self.approx_embedder(x)

        # Low rank embeddings are projected to contribute
        # to input, reset and update signals.
        # All the shapes: (n_words, dim)
        input_signals = []
        reset_signals = []
        update_signals = []
        for level in range(self.num_levels):
            input_signals.append(self.input_embedders[level](approx_embeddings))
            update_signals.append(self.update_embedders[level](approx_embeddings))
            reset_signals.append(self.reset_embedders[level](approx_embeddings))

        # Hidden layers.
        # Shape in case of matrix input: (max_seq_len, batch_size, dim)
        # Shape in case of vector input: (seq_len, dim)
        hidden_layers = []
        for level in range(self.num_levels):
            # Each hidden layer (except the bottom one) receives
            # input, reset and update signals from below.
            # All the shapes: (n_words, dim)
            if level > 0:
                input_signals[level] += self.inputers[level](hidden_layers[-1])
                update_signals[level] += self.updaters[level](hidden_layers[-1])
                reset_signals[level] += self.reseters[level](hidden_layers[-1])
            hidden_layers.append(self.transitions[level](
                    input_signals[level],
                    nsteps=x.shape[0],
                    batch_size=x.shape[1] if x.ndim == 2 else 1,
                    mask=x_mask,
                    gater_below=none_if_zero(update_signals[level]),
                    reseter_below=none_if_zero(reset_signals[level]),
                    use_noise=use_noise))
        if return_hidden_layers:
            print 'return hidden_layers'
            assert self.state['encoder_stack'] == 1
            return hidden_layers[0]

        # If we no stack of RNN but only a usual one,
        # then the last hidden state is used as a representation.
        # Return value shape in case of matrix input:
        #   (batch_size, dim)
        # Return value shape in case of vector input:
        #   (dim,)
        if self.num_levels == 1 or self.state['take_top']:
            print 'take_top'
            c = LastState()(hidden_layers[-1])
            if c.out.ndim == 2:
                c.out = c.out[:,:self.state['dim']]
            else:
                c.out = c.out[:self.state['dim']]
            return c

        # If we have a stack of RNN, then their last hidden states
        # are combined with a maxout layer.
        # Return value however has the same shape.
        contributions = []
        for level in range(self.num_levels):
            contributions.append(self.repr_contributors[level](
                LastState()(hidden_layers[level])))
        # I do not know a good starting value for sum
        c = self.repr_calculator(sum(contributions[1:], contributions[0]))
        return c

class NTMDecoder(NTMEncoderDecoderBase):

    EVALUATION = 0
    SAMPLING = 1
    BEAM_SEARCH = 2
    DEBUG = 3

    def __init__(self, state, rng, prefix='dec',
            skip_init=False, compute_alignment=False):
        self.state = state
        self.rng = rng
        self.prefix = prefix
        self.skip_init = skip_init
        self.compute_alignment = compute_alignment

        # Actually there is a problem here -
        # we don't make difference between number of input layers
        # and outputs layers.
        self.num_levels = self.state['decoder_stack']

        if 'dim_mult' not in self.state:
            self.state['dim_mult'] = 1.

    def create_layers(self):
        """ Create all elements of Decoder's computation graph"""

        self.default_kwargs = dict(
            init_fn=self.state['weight_init_fn'] if not self.skip_init else "sample_zeros",
            weight_noise=self.state['weight_noise'],
            scale=self.state['weight_scale'])

        self._create_embedding_layers()
        self._create_transition_layers(init_memory_param=False, head_fn = self.state['decoder_head_fn'])
        self._create_inter_level_layers()
        self._create_initialization_layers()
        self._create_decoding_layers()
        self._create_readout_layers()

        if self.state['search']:
            assert self.num_levels == 1
            self.transitions[0].set_decoding_layers(
                    self.decode_inputers[0],
                    self.decode_reseters[0],
                    self.decode_updaters[0])

    def _create_initialization_layers(self):
        logger.debug("_create_initialization_layers")
        self.initializers = [ZeroLayer()] * self.num_levels
        if self.state['bias_code']:
            for level in range(self.num_levels):
                self.initializers[level] = MultiLayer(
                    self.rng,
                    n_in=self.state['dim'],
                    n_hids=[self.state['dim'] * self.state['hid_mult']],
                    activation=[prefix_lookup(self.state, 'dec', 'activ')],
                    bias_scale=[self.state['bias']],
                    name='{}_initializer_{}'.format(self.prefix, level),
                    **self.default_kwargs)

    def _create_decoding_layers(self):
        logger.debug("_create_decoding_layers")
        self.decode_inputers = [lambda x : 0] * self.num_levels
        self.decode_reseters = [lambda x : 0] * self.num_levels
        self.decode_updaters = [lambda x : 0] * self.num_levels
        self.back_decode_inputers = [lambda x : 0] * self.num_levels
        self.back_decode_reseters = [lambda x : 0] * self.num_levels
        self.back_decode_updaters = [lambda x : 0] * self.num_levels

        decoding_kwargs = dict(self.default_kwargs)
        decoding_kwargs.update(dict(
                n_in=self.state['c_dim'],
                n_hids=self.state['dim'] * self.state['dim_mult'],
                activation=['lambda x:x'],
                learn_bias=False))

        if self.state['decoding_inputs']:
            for level in range(self.num_levels):
                # Input contributions
                self.decode_inputers[level] = MultiLayer(
                    self.rng,
                    name='{}_dec_inputter_{}'.format(self.prefix, level),
                    **decoding_kwargs)
                # Update gate contributions
                if prefix_lookup(self.state, 'dec', 'rec_gating'):
                    self.decode_updaters[level] = MultiLayer(
                        self.rng,
                        name='{}_dec_updater_{}'.format(self.prefix, level),
                        **decoding_kwargs)
                # Reset gate contributions
                if prefix_lookup(self.state, 'dec', 'rec_reseting'):
                    self.decode_reseters[level] = MultiLayer(
                        self.rng,
                        name='{}_dec_reseter_{}'.format(self.prefix, level),
                        **decoding_kwargs)

    def _create_readout_layers(self):
        softmax_layer = self.state['softmax_layer'] if 'softmax_layer' in self.state \
                        else 'SoftmaxLayer'

        logger.debug("_create_readout_layers")

        readout_kwargs = dict(self.default_kwargs)
        readout_kwargs.update(dict(
                n_hids=self.state['dim'],
                activation='lambda x: x',
            ))

        self.repr_readout = MultiLayer(
                self.rng,
                n_in=self.state['c_dim'],
                learn_bias=False,
                name='{}_repr_readout'.format(self.prefix),
                **readout_kwargs)

        # Attention - this is the only readout layer
        # with trainable bias. Should be careful with that.
        self.hidden_readouts = [None] * self.num_levels
        for level in range(self.num_levels):
            self.hidden_readouts[level] = MultiLayer(
                self.rng,
                n_in=self.state['dim'],
                name='{}_hid_readout_{}'.format(self.prefix, level),
                **readout_kwargs)

        self.prev_word_readout = 0
        if self.state['bigram']:
            self.prev_word_readout = MultiLayer(
                self.rng,
                n_in=self.state['rank_n_approx'],
                n_hids=self.state['dim'],
                activation=['lambda x:x'],
                learn_bias=False,
                name='{}_prev_readout_{}'.format(self.prefix, level),
                **self.default_kwargs)

        if self.state['deep_out']:
            act_layer = UnaryOp(activation=eval(self.state['unary_activ']))
            drop_layer = DropOp(rng=self.rng, dropout=self.state['dropout'])
            self.output_nonlinearities = [act_layer, drop_layer]
            self.output_layer = eval(softmax_layer)(
                    self.rng,
                    self.state['dim'] / self.state['maxout_part'],
                    self.state['n_sym_target'],
                    sparsity=-1,
                    rank_n_approx=self.state['rank_n_approx'],
                    name='{}_deep_softmax'.format(self.prefix),
                    use_nce=self.state['use_nce'] if 'use_nce' in self.state else False,
                    **self.default_kwargs)
        else:
            self.output_nonlinearities = []
            self.output_layer = eval(softmax_layer)(
                    self.rng,
                    self.state['dim'],
                    self.state['n_sym_target'],
                    sparsity=-1,
                    rank_n_approx=self.state['rank_n_approx'],
                    name='dec_softmax',
                    sum_over_time=True,
                    use_nce=self.state['use_nce'] if 'use_nce' in self.state else False,
                    **self.default_kwargs)

    def build_decoder(self, c, y,
            c_mask=None,
            y_mask=None,
            step_num=None,
            mode=EVALUATION,
            init_memories=None,
            given_init_states=None,
            given_init_memories=None,
            given_init_weights=None,
            T=1):
        """Create the computational graph of the RNN Decoder.

        :param c:
            representations produced by an encoder.
            (n_samples, dim) matrix if mode == sampling or
            (max_seq_len, batch_size, dim) matrix if mode == evaluation

        :param c_mask:
            if mode == evaluation a 0/1 matrix identifying valid positions in c

        :param y:
            if mode == evaluation
                target sequences, matrix of word indices of shape (max_seq_len, batch_size),
                where each column is a sequence
            if mode != evaluation
                a vector of previous words of shape (n_samples,)

        :param y_mask:
            if mode == evaluation a 0/1 matrix determining lengths
                of the target sequences, must be None otherwise

        :param mode:
            chooses on of three modes: evaluation, sampling and beam_search

        :param given_init_states:
            for sampling and beam_search. A list of hidden states
                matrices for each layer, each matrix is (n_samples, dim)

        :param T:
            sampling temperature
        """

        assert self.num_levels == 1
        # Check parameter consistency
        if mode == NTMDecoder.EVALUATION:
            assert not given_init_states
        else:
            assert not y_mask
            assert given_init_states
            if mode == NTMDecoder.BEAM_SEARCH:
                assert T == 1

        # For log-likelihood evaluation the representation
        # be replicated for conveniency. In case backward RNN is used
        # it is not done.
        # Shape if mode == evaluation
        #   (max_seq_len, batch_size, dim)
        # Shape if mode != evaluation
        #   (n_samples, dim)
        if not self.state['search']:
            if mode == NTMDecoder.EVALUATION:
                c = PadLayer(y.shape[0])(c)
            else:
                assert step_num
                c_pos = TT.minimum(step_num, c.shape[0] - 1)

        # Low rank embeddings of all the input words.
        # Shape if mode == evaluation
        #   (n_words, rank_n_approx),
        # Shape if mode != evaluation
        #   (n_samples, rank_n_approx)
        approx_embeddings = self.approx_embedder(y)

        # Low rank embeddings are projected to contribute
        # to input, reset and update signals.
        # All the shapes if mode == evaluation:
        #   (n_words, dim)
        # where: n_words = max_seq_len * batch_size
        # All the shape if mode != evaluation:
        #   (n_samples, dim)
        input_signals = []
        reset_signals = []
        update_signals = []
        for level in range(self.num_levels):
            # Contributions directly from input words.
            input_signals.append(self.input_embedders[level](approx_embeddings))
            update_signals.append(self.update_embedders[level](approx_embeddings))
            reset_signals.append(self.reset_embedders[level](approx_embeddings))

            # Contributions from the encoded source sentence.
            if not self.state['search'] and self.state['c_weight'] != 0. :
                print 'contribute from encoded source sentences' 
                current_c = c if mode == NTMDecoder.EVALUATION else c[c_pos]
                input_signals[level] += self.decode_inputers[level](current_c)
                update_signals[level] += self.decode_updaters[level](current_c)
                reset_signals[level] += self.decode_reseters[level](current_c)

        # Hidden layers' initial states.
        # Shapes if mode == evaluation:
        #   (batch_size, dim)
        # Shape if mode != evaluation:
        #   (n_samples, dim)
        init_states = given_init_states
        if not init_states:
            init_states = []
            for level in range(self.num_levels):
                init_c = c[0, :, -self.state['dim']:]
                init_states.append(self.initializers[level](init_c))

        # Hidden layers' states.
        # Shapes if mode == evaluation:
        #  (seq_len, batch_size, dim)
        # Shapes if mode != evaluation:
        #  (n_samples, dim)
        hidden_layers = []
        contexts = []
        # Default value for alignment must be smth computable
        alignment = TT.zeros((1,))
        for level in range(self.num_levels):
            if level > 0:
                input_signals[level] += self.inputers[level](hidden_layers[level - 1])
                update_signals[level] += self.updaters[level](hidden_layers[level - 1])
                reset_signals[level] += self.reseters[level](hidden_layers[level - 1])
            if self.state['dec_rec_layer'] == 'NTMLayer' or self.state['dec_rec_layer'] == 'NTMLayerWithSearch':
                add_kwargs = (dict(state_before=init_states[level],
                                memory_before=given_init_memories)
                        if mode != NTMDecoder.EVALUATION
                        else dict(init_state=init_states[level],
                            init_memory=init_memories,
                            batch_size=y.shape[1] if y.ndim == 2 else 1,
                            nsteps=y.shape[0]))
            else:
                add_kwargs = (dict(state_before=init_states[level])
                        if mode != NTMDecoder.EVALUATION
                        else dict(init_state=init_states[level],
                            batch_size=y.shape[1] if y.ndim == 2 else 1,
                            nsteps=y.shape[0]))
            if self.state['search']:
                add_kwargs['c'] = c
                add_kwargs['c_mask'] = c_mask
                add_kwargs['return_alignment'] = self.compute_alignment
                if mode != NTMDecoder.EVALUATION:
                    add_kwargs['step_num'] = step_num
            result = self.transitions[level](
                    input_signals[level],
                    mask=y_mask,
                    gater_below=none_if_zero(update_signals[level]),
                    reseter_below=none_if_zero(reset_signals[level]),
                    one_step=mode != NTMDecoder.EVALUATION,
                    use_noise=mode == NTMDecoder.EVALUATION,
                    **add_kwargs)
            if self.state['search']:
                if self.state['dec_rec_layer'] == 'NTMLayer' or self.state['dec_rec_layer'] == 'NTMLayerWithSearch':
                    if self.compute_alignment:
                        #This implicitly wraps each element of result.out with a Layer to keep track of the parameters.
                        #It is equivalent to h=result[0], ctx=result[1] etc. 
                        h, mem, rw,ww, ctx, alignment = result
                        if mode == NTMDecoder.EVALUATION:
                            alignment = alignment.out
                    else:
                        #This implicitly wraps each element of result.out with a Layer to keep track of the parameters.
                        #It is equivalent to h=result[0], ctx=result[1]
                        h, mem, rw,ww, ctx= result
                else:
                    if self.compute_alignment:
                        h, ctx, alignment = result
                        if mode == NTMDecoder.EVALUATION:
                            alignment = alignment.out
                    else:
                        h, ctx= result
            else:
                if self.state['dec_rec_layer'] == 'NTMLayer' or self.state['dec_rec_layer'] == 'NTMLayerWithSearch':
                    h,mem,rw,ww= result
                else:
                    h = result
                if mode == NTMDecoder.EVALUATION:
                    ctx = c
                else:
                    ctx = ReplicateLayer(given_init_states[0].shape[0])(c[c_pos]).out
            hidden_layers.append(h)
            contexts.append(ctx)

        # In hidden_layers we do no have the initial state, but we need it.
        # Instead of it we have the last one, which we do not need.
        # So what we do is discard the last one and prepend the initial one.
        if mode == NTMDecoder.EVALUATION:
            for level in range(self.num_levels):
                hidden_layers[level].out = TT.concatenate([
                    TT.shape_padleft(init_states[level].out),
                        hidden_layers[level].out])[:-1]

        # The output representation to be fed in softmax.
        # Shape if mode == evaluation
        #   (n_words, dim_r)
        # Shape if mode != evaluation
        #   (n_samples, dim_r)
        # ... where dim_r depends on 'deep_out' option.
        readout = self.repr_readout(contexts[0])
        for level in range(self.num_levels):
            if mode != NTMDecoder.EVALUATION:
                read_from = init_states[level]
            else:
                read_from = hidden_layers[level]
            read_from_var = read_from if type(read_from) == theano.tensor.TensorVariable else read_from.out
            if read_from_var.ndim == 3:
                read_from_var = read_from_var[:,:,:self.state['dim']]
            else:
                read_from_var = read_from_var[:,:self.state['dim']]
            if type(read_from) != theano.tensor.TensorVariable:
                read_from.out = read_from_var
            else:
                read_from = read_from_var
            readout += self.hidden_readouts[level](read_from)
        if self.state['bigram']:
            if mode != NTMDecoder.EVALUATION:
                check_first_word = (y > 0
                    if self.state['check_first_word']
                    else TT.ones((y.shape[0]), dtype="float32"))
                # padright is necessary as we want to multiply each row with a certain scalar
                readout += TT.shape_padright(check_first_word) * self.prev_word_readout(approx_embeddings).out
            else:
                if y.ndim == 1:
                    readout += Shift()(self.prev_word_readout(approx_embeddings).reshape(
                        (y.shape[0], 1, self.state['dim'])))
                else:
                    # This place needs explanation. When prev_word_readout is applied to
                    # approx_embeddings the resulting shape is
                    # (n_batches * sequence_length, repr_dimensionality). We first
                    # transform it into 3D tensor to shift forward in time. Then
                    # reshape it back.
                    readout += Shift()(self.prev_word_readout(approx_embeddings).reshape(
                        (y.shape[0], y.shape[1], self.state['dim']))).reshape(
                                readout.out.shape)
        for fun in self.output_nonlinearities:
            readout = fun(readout)

        if mode == NTMDecoder.SAMPLING:
            sample = self.output_layer.get_sample(
                    state_below=readout,
                    temp=T)
            # Current SoftmaxLayer.get_cost is stupid,
            # that's why we have to reshape a lot.
            self.output_layer.get_cost(
                    state_below=readout.out,
                    temp=T,
                    target=sample)
            log_prob = self.output_layer.cost_per_sample
            if self.state['dec_rec_layer'] == 'NTMLayer' or self.state['dec_rec_layer'] == 'NTMLayerWithSearch':
                return [sample] + [log_prob] + hidden_layers +[mem]
            else:
                return [sample] + [log_prob] + hidden_layers
        elif mode == NTMDecoder.BEAM_SEARCH:
            return self.output_layer(
                    state_below=readout.out,
                    temp=T).out
        elif mode == NTMDecoder.EVALUATION:
            return (self.output_layer.train(
                    state_below=readout,
                    target=y,
                    mask=y_mask,
                    reg=None),
                    alignment)
        elif mode == NTMDecoder.DEBUG:
            return [h, mem,rw,ww]
        else:
            raise Exception("Unknown mode for build_decoder")


    def sampling_step(self, *args):
        """Implements one step of sampling

        Args are necessary since the number (and the order) of arguments can vary"""

        args = iter(args)

        # Arguments that correspond to scan's "sequences" parameteter:
        step_num = next(args)
        assert step_num.ndim == 0

        # Arguments that correspond to scan's "outputs" parameteter:
        prev_word = next(args)
        assert prev_word.ndim == 1
        # skip the previous word log probability
        assert next(args).ndim == 1
        prev_hidden_states = [next(args) for k in range(self.num_levels)]
        assert prev_hidden_states[0].ndim == 2
        if self.state['dec_rec_layer'] == 'NTMLayer' or self.state['dec_rec_layer'] == 'NTMLayerWithSearch':
            prev_memory = next(args)

        # Arguments that correspond to scan's "non_sequences":
        c = next(args)
        assert c.ndim == 2
        T = next(args)
        assert T.ndim == 0

        if self.state['dec_rec_layer'] == 'NTMLayer' or self.state['dec_rec_layer'] == 'NTMLayerWithSearch':
            decoder_args = dict(given_init_states=prev_hidden_states,
                            given_init_memories=prev_memory,
                            #given_init_weights=prev_weights,
                             T=T, c=c)
        else:
            decoder_args = dict(given_init_states=prev_hidden_states, T=T, c=c)

        sample, log_prob = self.build_decoder(y=prev_word, step_num=step_num, mode=NTMDecoder.SAMPLING, **decoder_args)[:2]
        if self.state['dec_rec_layer'] == 'NTMLayer' or self.state['dec_rec_layer'] == 'NTMLayerWithSearch':
            hidden_states, mem = self.build_decoder(y=sample, step_num=step_num, mode=NTMDecoder.SAMPLING, **decoder_args)[2:]
            return [sample, log_prob,hidden_states,mem]
        else:
            hidden_states = self.build_decoder(y=sample, step_num=step_num, mode=NTMDecoder.SAMPLING, **decoder_args)[2:]
            return [sample, log_prob] + hidden_states

    def build_initializers(self, c):
        return [init(c).out for init in self.initializers]

    def build_sampler(self, n_samples, n_steps, T, c, m = None):
        states = [TT.zeros(shape=(n_samples,), dtype='int64'),
                TT.zeros(shape=(n_samples,), dtype='float32')]
        init_c = c[0, -self.state['dim']:]
        states += [ReplicateLayer(n_samples)(init(init_c).out).out for init in self.initializers]
        if m:
            states += [ReplicateLayer(n_samples)(m).out]
        #states += [ReplicateLayer(n_samples)(init(init_c).out).out for init in self.initializers]

        if not self.state['search']:
            c = PadLayer(n_steps)(c).out

        # Pad with final states
        non_sequences = [c, T]

        outputs, updates = theano.scan(self.sampling_step,
                outputs_info=states,
                non_sequences=non_sequences,
                sequences=[TT.arange(n_steps, dtype="int64")],
                n_steps=n_steps,
                name="{}_sampler_scan".format(self.prefix))
        return (outputs[0], outputs[1]), updates

    def build_next_probs_predictor(self, c, step_num, y, init_states, m=None):
        return self.build_decoder(c, y, mode=NTMDecoder.BEAM_SEARCH,
                given_init_states=init_states, step_num=step_num, given_init_memories=m)

    def build_next_states_computer(self, c, step_num, y, init_states, m=None):
        return self.build_decoder(c, y, mode=NTMDecoder.SAMPLING,
                given_init_states=init_states, step_num=step_num, given_init_memories=m)[2:]

    def build_next_debug_computer(self, c, step_num, y, init_states, m=None):
        return self.build_decoder(c, y, mode=NTMDecoder.DEBUG,
                given_init_states=init_states, step_num=step_num, given_init_memories=m)

class NTMEncoderDecoder(object):
    """This class encapsulates the translation model.

    The expected usage pattern is:
    >>> encdec = RNNEncoderDecoder(...)
    >>> encdec.build(...)
    >>> useful_smth = encdec.create_useful_smth(...)

    Functions from the create_smth family (except create_lm_model)
    when called complile and return functions that do useful stuff.
    """

    def __init__(self, state, rng,
            skip_init=False,
            compute_alignment=False):
        """Constructor.

        :param state:
            A state in the usual groundhog sense.
        :param rng:
            Random number generator. Something like numpy.random.RandomState(seed).
        :param skip_init:
            If True, all the layers are initialized with zeros. Saves time spent on
            parameter initialization if they are loaded later anyway.
        :param compute_alignment:
            If True, the alignment is returned by the decoder.
        """

        self.state = state
        self.rng = rng
        self.skip_init = skip_init
        self.compute_alignment = compute_alignment
        if not self.state.has_key('decoder_head_fn'):
            self.state['decoder_head_fn'] = self.state['head_fn']
        if not self.state.has_key('encoder_head_fn'):
            self.state['encoder_head_fn'] = self.state['head_fn']
        if not self.state.has_key('c_weight'):
            self.state['c_weight'] = 1.
        if not self.state.has_key('encoder_memory_weight'):
            self.state['encoder_memory_weight'] = self.state['memory_weight']
        if not self.state.has_key('decoder_memory_weight'):
            self.state['decoder_memory_weight'] = self.state['memory_weight']
        if not self.state.has_key('encoder_memory_param'):
            self.state['encoder_memory_param'] = True

    def build(self):
        logger.debug("Create input variables")
        self.x = TT.lmatrix('x')
        self.x_mask = TT.matrix('x_mask')
        self.y = TT.lmatrix('y')
        self.y_mask = TT.matrix('y_mask')
        self.inputs = [self.x, self.y, self.x_mask, self.y_mask]

        # Annotation for the log-likelihood computation
        training_c_components = []

        logger.debug("Create encoder")
        self.encoder = NTMEncoder(self.state, self.rng,
                prefix="enc",
                skip_init=self.skip_init)
        self.encoder.create_layers()

        logger.debug("Build encoding computation graph")
        forward_training = self.encoder.build_encoder(
                self.x, self.x_mask,
                use_noise=True,
                return_hidden_layers=True)
        forward_training_c, forward_training_m, forward_training_rw, forward_training_ww= forward_training

        logger.debug("Create backward encoder")
        self.backward_encoder = NTMEncoder(self.state, self.rng,
                prefix="back_enc",
                skip_init=self.skip_init)
        self.backward_encoder.create_layers()

        logger.debug("Build backward encoding computation graph")
        backward_training = self.backward_encoder.build_encoder(
                self.x[::-1],
                self.x_mask[::-1],
                use_noise=True,
                approx_embeddings=self.encoder.approx_embedder(self.x[::-1]),
                return_hidden_layers=True)
        backward_training_c, backward_training_m, backward_training_rw, backward_training_ww = backward_training
        # Reverse time for backward representations.
        backward_training_c.out = backward_training_c.out[::-1]

        self.forward_training = forward_training
        self.forward_training_c = forward_training_c
        self.forward_training_m = forward_training_m
        self.forward_training_rw = forward_training_rw
        self.backward_training = backward_training
        self.backward_training_c = backward_training_c
        self.backward_training_m = backward_training_m
        self.backward_training_rw = backward_training_rw


        if self.state['forward']:
            print 'forward'
            training_c_components.append(forward_training_c)
        if self.state['last_forward']:
            print 'last_forward'
            training_c_components.append(
                    ReplicateLayer(self.x.shape[0])(forward_training_c[-1]))
        if self.state['backward']:
            print 'backward'
            training_c_components.append(backward_training_c)
        if self.state['last_backward']:
            print 'last_backward'
            training_c_components.append(ReplicateLayer(self.x.shape[0])
                    (backward_training_c[0]))
        self.state['c_dim'] = len(training_c_components) * self.state['dim']

        self.training_c = Concatenate(axis=2)(*training_c_components)

        logger.debug("Create decoder")
        self.decoder = NTMDecoder(self.state, self.rng,
                skip_init=self.skip_init, compute_alignment=self.compute_alignment)
        self.decoder.create_layers()
        logger.debug("Build log-likelihood computation graph")
        ini_training_mem = None
        if self.state['dec_rec_layer'] == 'NTMLayer':
            print 'decoder training with memory'
            ini_training_mem = self.forward_training_m[-1]
        if self.state['dec_rec_layer'] == 'NTMLayerWithSearch':
            print 'decoder training with backward memory'
            ini_training_mem = self.backward_training_m[-1]
        self.predictions, self.alignment = self.decoder.build_decoder(
                c=Concatenate(axis=2)(*training_c_components), c_mask=self.x_mask,
                y=self.y, y_mask=self.y_mask,
                init_memories=ini_training_mem)

        # Annotation for sampling
        sampling_c_components = []

        logger.debug("Build sampling computation graph")
        self.sampling_x = TT.lvector("sampling_x")
        self.n_samples = TT.lscalar("n_samples")
        self.n_steps = TT.lscalar("n_steps")
        self.T = TT.scalar("T")
        self.forward_sampling = self.encoder.build_encoder(
                self.sampling_x,
                return_hidden_layers=True)
        self.forward_sampling_c, self.forward_sampling_m, self.forward_sampling_rw, self.forward_sampling_ww = self.forward_sampling.out
        self.backward_sampling = self.backward_encoder.build_encoder(
                self.sampling_x[::-1],
                approx_embeddings=self.encoder.approx_embedder(self.sampling_x[::-1]),
                return_hidden_layers=True)
        self.backward_sampling_c_reverse, self.backward_sampling_m,self.backward_sampling_rw,self.backward_sampling_ww = self.backward_sampling.out
        self.backward_sampling_c = self.backward_sampling_c_reverse[::-1]
        if self.state['forward']:
            print 'forward'
            sampling_c_components.append(self.forward_sampling_c)
        if self.state['last_forward']:
            print 'last_forward'
            sampling_c_components.append(ReplicateLayer(self.sampling_x.shape[0])
                    (self.forward_sampling_c[-1]))
        if self.state['backward']:
            print 'backward'
            sampling_c_components.append(self.backward_sampling_c)
        if self.state['last_backward']:
            print 'last_backward'
            sampling_c_components.append(ReplicateLayer(self.sampling_x.shape[0])
                    (self.backward_sampling_c[0]))

        self.sampling_c_components = sampling_c_components
        print 'scc:',len(sampling_c_components)
        ini_sampling_mem = None
        if self.state['dec_rec_layer'] == 'NTMLayer':
            print 'decoder sampling with memory'
            ini_sampling_mem = self.forward_sampling_m[-1]
            self.return_sampling_mem = self.forward_sampling_m
        if self.state['dec_rec_layer'] == 'NTMLayerWithSearch':
            print 'decoder sampling with backward memory'
            ini_sampling_mem = self.backward_sampling_m[-1]
            self.return_sampling_mem = self.backward_sampling_m
        self.sampling_c = Concatenate(axis=1)(*sampling_c_components).out
        (self.sample, self.sample_log_prob), self.sampling_updates =\
            self.decoder.build_sampler(self.n_samples, self.n_steps, self.T,
                    c=self.sampling_c,
                    m=ini_sampling_mem)

        logger.debug("Create auxiliary variables")
        self.c = TT.matrix("c")
        self.step_num = TT.lscalar("step_num")
        self.current_states = [TT.matrix("cur_{}".format(i))
                for i in range(self.decoder.num_levels)]
        self.current_memory = TT.tensor3("curm")
        self.gen_y = TT.lvector("gen_y")

    def create_lm_model(self):
        if hasattr(self, 'lm_model'):
            return self.lm_model
        self.lm_model = LM_Model(
            cost_layer=self.predictions,
            sample_fn=self.create_sampler(),
            weight_noise_amount=self.state['weight_noise_amount'],
            indx_word=self.state['indx_word_target'],
            indx_word_src=self.state['indx_word'],
            rng=self.rng)
        self.lm_model.load_dict(self.state)
        logger.debug("Model params:\n{}".format(
            pprint.pformat(sorted([p.name for p in self.lm_model.params]))))
        return self.lm_model

    def create_representation_computer(self):
        ou = [self.sampling_c]
        if self.state['dec_rec_layer'] == 'NTMLayer' or self.state['dec_rec_layer'] == 'NTMLayerWithSearch':
            ou = [self.sampling_c, self.return_sampling_mem]
        if not hasattr(self, "repr_fn"):
            self.repr_fn = theano.function(
                    inputs=[self.sampling_x],
                    outputs=ou,
                    name="repr_fn")
        return self.repr_fn

    def create_initializers(self):
        if not hasattr(self, "init_fn"):
            init_c = self.sampling_c[0, -self.state['dim']:]
            self.init_fn = theano.function(
                    inputs=[self.sampling_c],
                    outputs=self.decoder.build_initializers(init_c),
                    name="init_fn")
        return self.init_fn

    def create_sampler(self, many_samples=False):
        if hasattr(self, 'sample_fn'):
            return self.sample_fn
        logger.debug("Compile sampler")
        self.sample_fn = theano.function(
                inputs=[self.n_samples, self.n_steps, self.T, self.sampling_x],
                outputs=[self.sample, self.sample_log_prob],
                updates=self.sampling_updates,
                name="sample_fn")
        if not many_samples:
            def sampler(*args):
                return map(lambda x : x.squeeze(), self.sample_fn(1, *args))
            return sampler
        return self.sample_fn

    def view_encoder_weight(self):
        self.encoder_weight_fn = theano.function(
                        inputs = [self.sampling_x],
                        outputs = [self.forward_sampling_rw,self.forward_sampling_ww])
        return self.encoder_weight_fn
        
    def create_next_debug_computer(self):
        if not hasattr(self, 'next_debug_fn'):
            if self.state['dec_rec_layer'] == 'NTMLayer' or self.state['dec_rec_layer'] == 'NTMLayerWithSearch':
                self.next_probs_fn = theano.function(
                    inputs=[self.c, self.step_num, self.gen_y, self.current_memory] + self.current_states,
                    outputs=self.decoder.build_next_debug_computer(
                        self.c, self.step_num, self.gen_y, self.current_states,m=self.current_memory),
                    name="next_debug_fn")
            else:
                self.next_probs_fn = theano.function(
                    inputs=[self.c, self.step_num, self.gen_y] + self.current_states,
                    outputs=self.decoder.build_next_debug_computer(
                        self.c, self.step_num, self.gen_y, self.current_states),
                    name="next_debug_fn")
        return self.next_probs_fn

    def create_scorer(self, batch=False):
        if not hasattr(self, 'score_fn'):
            logger.debug("Compile scorer")
            self.score_fn = theano.function(
                    inputs=self.inputs,
                    outputs=[-self.predictions.cost_per_sample],
                    name="score_fn")
        if batch:
            return self.score_fn
        def scorer(x, y):
            x_mask = numpy.ones(x.shape[0], dtype="float32")
            y_mask = numpy.ones(y.shape[0], dtype="float32")
            return self.score_fn(x[:, None], y[:, None],
                    x_mask[:, None], y_mask[:, None])
        return scorer

    def create_next_probs_computer(self):
        if not hasattr(self, 'next_probs_fn'):
            if self.state['dec_rec_layer'] == 'NTMLayer' or self.state['dec_rec_layer'] == 'NTMLayerWithSearch':
                self.next_probs_fn = theano.function(
                    inputs=[self.c, self.step_num, self.gen_y, self.current_memory] + self.current_states,
                    outputs=[self.decoder.build_next_probs_predictor(
                        self.c, self.step_num, self.gen_y, self.current_states,m=self.current_memory)],
                    name="next_probs_fn")
            else:
                self.next_probs_fn = theano.function(
                    inputs=[self.c, self.step_num, self.gen_y] + self.current_states,
                    outputs=[self.decoder.build_next_probs_predictor(
                        self.c, self.step_num, self.gen_y, self.current_states)],
                    name="next_probs_fn")
        return self.next_probs_fn

    def create_next_states_computer(self):
        if not hasattr(self, 'next_states_fn'):
            if self.state['dec_rec_layer'] == 'NTMLayer' or self.state['dec_rec_layer'] == 'NTMLayerWithSearch':
                self.next_states_fn = theano.function(
                    inputs=[self.c, self.step_num, self.gen_y, self.current_memory] + self.current_states,
                    outputs=self.decoder.build_next_states_computer(
                        self.c, self.step_num, self.gen_y, self.current_states,m=self.current_memory),
                    name="next_states_fn")
            else:
                self.next_states_fn = theano.function(
                    inputs=[self.c, self.step_num, self.gen_y] + self.current_states,
                    outputs=self.decoder.build_next_states_computer(
                        self.c, self.step_num, self.gen_y, self.current_states),
                    name="next_states_fn")
        return self.next_states_fn


    def create_probs_computer(self, return_alignment=False):
        if not hasattr(self, 'probs_fn'):
            logger.debug("Compile probs computer")
            self.probs_fn = theano.function(
                    inputs=self.inputs,
                    outputs=[self.predictions.word_probs, self.alignment],
                    name="probs_fn")
        def probs_computer(x, y):
            x_mask = numpy.ones(x.shape[0], dtype="float32")
            y_mask = numpy.ones(y.shape[0], dtype="float32")
            probs, alignment = self.probs_fn(x[:, None], y[:, None],
                    x_mask[:, None], y_mask[:, None])
            if return_alignment:
                return probs, alignment
            else:
                return probs
        return probs_computer
