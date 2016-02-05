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
        parse_input
import groundhog.utils as utils

logger = logging.getLogger(__name__)

def cosine_sim(k,M):
    #print k.ndim
    k_unit = k / ( TT.sqrt(TT.sum(k**2)) + 1e-5 )
    k_unit = k_unit.dimshuffle(('x',0)) #T.patternbroadcast(k_unit.reshape((1,k_unit.shape[0])),(True,False))
    k_unit.name = "k_unit"
    M_lengths = TT.sqrt(TT.sum(M**2,axis=1)).dimshuffle((0,'x'))
    M_unit = M / ( M_lengths + 1e-5 )
    
    M_unit.name = "M_unit"
#   M_unit = Print("M_unit")(M_unit)
    return TT.sum(k_unit * M_unit,axis=1)

def cosine_sim_batch(k, M):
    k_lengths = TT.sqrt(TT.sum(k**2,axis=1)).dimshuffle((0,'x'))
    k_unit = k / ( k_lengths + 1e-5 )
    k_unit = k_unit.dimshuffle((0,'x',1)) #T.patternbroadcast(k_unit.reshape((1,k_unit.shape[0])),(True,False))
    k_unit.name = "k_unit"
    M_lengths = TT.sqrt(TT.sum(M**2,axis=2)).dimshuffle((0,1,'x'))
    M_unit = M / ( M_lengths + 1e-5 )
    return TT.sum(k_unit * M_unit,axis=2)

class NTMLayerBase(Layer):

    def __init__(self, n_hids, n_hids2, rng,name):
        super(NTMLayerBase, self).__init__(n_hids, n_hids2, rng,name)

    def write_normalhead_process(self,h,weight_before,memory_before):
        key = TT.dot(h, self.head[0]['W_key'])+self.head[0]['b_key']
        beta = TT.nnet.softplus(TT.dot(h, self.head[0]['W_beta'])+self.head[0]['b_beta'])
        g = TT.nnet.sigmoid(TT.dot(h, self.head[0]['W_g'])+self.head[0]['b_g'])
        add = TT.dot(h, self.head[0]['W_add'])+self.head[0]['b_add']
        erase = TT.nnet.sigmoid(TT.dot(h, self.head[0]['W_erase'])+self.head[0]['b_erase'])
        print key.ndim
        print beta.ndim
        print g.ndim
        print add.ndim
        print erase.ndim
        sim = cosine_sim(key, memory_before)
        weight_c = TT.nnet.softmax(beta.reshape((1,))*sim)
        g = g.reshape((1,))
        weight_g = g*weight_c+(1-g)*weight_before
        weight = weight_g
        weight = weight.reshape((weight.shape[1],))
        weight_dim = weight.dimshuffle((0, 'x'))
        erase_dim = erase.reshape((erase.shape[0],)).dimshuffle(('x', 0))
        add_dim = add.reshape((add.shape[0],)).dimshuffle(('x', 0))
        memory_erase = memory_before*(1-(weight_dim*erase_dim))
        memory = memory_erase+(weight_dim*add_dim)
        return weight, memory

    def write_normalhead_process_batch(self,h,weight_before,memory_before):
        key = TT.dot(h, self.head[0]['W_key'])+self.head[0]['b_key']
        beta = TT.nnet.softplus(TT.dot(h, self.head[0]['W_beta'])+self.head[0]['b_beta'])
        g = TT.nnet.sigmoid(TT.dot(h, self.head[0]['W_g'])+self.head[0]['b_g'])
        add = TT.dot(h, self.head[0]['W_add'])+self.head[0]['b_add']
        erase = TT.nnet.sigmoid(TT.dot(h, self.head[0]['W_erase'])+self.head[0]['b_erase'])
        print key.ndim
        print beta.ndim
        print g.ndim
        print add.ndim
        print erase.ndim
        sim = cosine_sim_batch(key,memory_before)
        weight_c = TT.nnet.softmax(beta.reshape((beta.shape[0],)).dimshuffle(0,'x')*sim)
        g = g.reshape((g.shape[0],)).dimshuffle(0,'x')
        weight_g = g*weight_c+(1-g)*weight_before
        weight = weight_g
        weight_dim = weight.dimshuffle((0, 1,'x'))
        erase_dim = erase.dimshuffle((0,'x', 1))
        add_dim = add.dimshuffle((0,'x', 1))
        memory_erase = memory_before*(1-(weight_dim*erase_dim))
        memory = memory_erase+(weight_dim*add_dim)
        return weight, memory

    def init_head(self, num):

        logger.debug('normal head num: %d', num)
        self.head = [{}]
        for i in range(num):
            self.head[i]['W_input'] = theano.shared(
                                        self.memory_init_fn(self.memory_dim,
                                            self.n_hids,
                                            self.sparsity,
                                            self.scale,
                                            self.rng), 
                                            name='W_head_input_%s'%self.name)
            self.head[i]['W_reset'] = theano.shared(
                                        self.memory_init_fn(self.memory_dim,
                                            self.n_hids,
                                            self.sparsity,
                                            self.scale,
                                            self.rng), 
                                            name='W_head_reset_%s'%self.name)
            self.head[i]['W_update'] = theano.shared(
                                        self.memory_init_fn(self.memory_dim,
                                            self.n_hids,
                                            self.sparsity,
                                            self.scale,
                                            self.rng), 
                                            name='W_head_update_%s'%self.name)
            self.params.append(self.head[i]['W_input'])
            self.params.append(self.head[i]['W_reset'])
            self.params.append(self.head[i]['W_update'])
            self.head[i]['W_key'] = theano.shared(
                                        self.memory_init_fn(self.n_hids,
                                            self.memory_dim,
                                            self.sparsity,
                                            self.scale,
                                            self.rng), 
                                            name='W_head_key_%s'%self.name)
            self.head[i]['b_key'] = theano.shared(
                                        self.bias_fn(self.memory_dim,self.bias_scale,rng=self.rng),
                                        name="b_head_key_%s"%self.name)
            self.head[i]['W_beta'] = theano.shared(
                                        self.memory_init_fn(self.n_hids,
                                            1,
                                            self.sparsity,
                                            self.scale,
                                            self.rng), 
                                            name='W_head_beta_%s'%self.name)
            self.head[i]['b_beta'] = theano.shared(
                                        self.bias_fn(1,self.bias_scale,rng=self.rng),
                                        name="b_head_beta_%s"%self.name)
            self.head[i]['W_g'] = theano.shared(
                                        self.memory_init_fn(self.n_hids,
                                            1,
                                            self.sparsity,
                                            self.scale,
                                            self.rng), 
                                            name='W_head_g_%s'%self.name)
            self.head[i]['b_g'] = theano.shared(
                                        self.bias_fn(1,self.bias_scale,rng=self.rng),
                                        name="b_head_g_%s"%self.name)
            self.head[i]['W_shift'] = theano.shared(
                                        self.init_fn(self.memory_size,
                                            self.memory_size,
                                            self.sparsity,
                                            self.scale,
                                            self.rng), 
                                            name='W_head_location_%s'%self.name)
            self.head[i]['b_shift'] = theano.shared(
                                        self.bias_fn(self.memory_size,self.bias_scale,rng=self.rng),
                                        name="b_head_location_%s"%self.name)
            self.head[i]['W_gamma'] = theano.shared(
                                        self.init_fn(self.memory_size,
                                            self.memory_size,
                                            self.sparsity,
                                            self.scale,
                                            self.rng), 
                                            name='W_head_gamma_%s'%self.name)
            self.head[i]['b_gamma'] = theano.shared(
                                        self.bias_fn(self.memory_size,self.bias_scale,rng=self.rng),
                                        name="b_head_gamma_%s"%self.name)
            self.head[i]['W_erase'] = theano.shared(
                                        self.memory_init_fn(self.n_hids,
                                            self.memory_dim,
                                            self.sparsity,
                                            self.scale,
                                            self.rng), 
                                            name='W_head_erase_%s'%self.name)
            self.head[i]['b_erase'] = theano.shared(
                                        self.bias_fn(self.memory_dim,self.bias_scale,rng=self.rng),
                                        name="b_head_erase_%s"%self.name)
            self.head[i]['W_add'] = theano.shared(
                                        self.memory_init_fn(self.n_hids,
                                            self.memory_dim,
                                            self.sparsity,
                                            self.scale,
                                            self.rng), 
                                            name='W_head_add_%s'%self.name)
            self.head[i]['b_add'] = theano.shared(
                                        self.bias_fn(self.memory_dim,self.bias_scale,rng=self.rng),
                                        name="b_head_add_%s"%self.name)
            
            self.params.append(self.head[i]['W_key'])
            self.params.append(self.head[i]['b_key'])
            self.params.append(self.head[i]['W_beta'])
            self.params.append(self.head[i]['b_beta'])
            
            self.params.append(self.head[i]['W_g'])
            self.params.append(self.head[i]['b_g'])
            '''
            self.params.append(self.head[i]['W_shift'])
            self.params.append(self.head[i]['b_shift'])
            self.params.append(self.head[i]['W_gamma'])
            self.params.append(self.head[i]['b_gamma'])
            '''
            self.params.append(self.head[i]['W_erase'])
            self.params.append(self.head[i]['b_erase'])
            self.params.append(self.head[i]['W_add'])
            self.params.append(self.head[i]['b_add'])
        self.head_process = self.normalhead_process
        self.head_process_batch = self.normalhead_process_batch

    def read_neuralhead_process(self,h,weight_before,memory_before):
        key = TT.dot(h, self.head[0]['W_readkey'])+self.head[0]['b_readkey']
        beta = TT.nnet.softplus(TT.dot(h, self.head[0]['W_readbeta'])+self.head[0]['b_readbeta'])
        print key.ndim
        print beta.ndim
        sim = cosine_sim(key, memory_before)
        weight_c = TT.nnet.softmax(beta.reshape((1,))*sim)
        weight_shift = TT.dot(weight_c, self.head[0]['W_readsim'])+TT.dot(weight_before, self.head[0]['W_readold'])+self.head[0]['b_readold']
        weight_shift = TT.nnet.softmax(weight_shift)
        weight = weight_shift
        weight = weight.reshape((weight.shape[1],))
        return weight

    def read_neuralhead_process_batch(self,h,weight_before,memory_before):
        key = TT.dot(h, self.head[0]['W_readkey'])+self.head[0]['b_readkey']
        beta = TT.nnet.softplus(TT.dot(h, self.head[0]['W_readbeta'])+self.head[0]['b_readbeta'])
        print key.ndim
        print beta.ndim
        sim = cosine_sim_batch(key,memory_before)
        weight_c = TT.nnet.softmax(beta.reshape((beta.shape[0],)).dimshuffle(0,'x')*sim)
        weight_shift = TT.dot(weight_c, self.head[0]['W_readsim'])+TT.dot(weight_before, self.head[0]['W_readold'])+self.head[0]['b_readold']
        weight_shift = TT.nnet.softmax(weight_shift)
        weight = weight_shift
        return weight

    def write_neuralhead_process(self,h,weight_before,memory_before):
        key = TT.dot(h, self.head[0]['W_key'])+self.head[0]['b_key']
        beta = TT.nnet.softplus(TT.dot(h, self.head[0]['W_beta'])+self.head[0]['b_beta'])
        add = TT.dot(h, self.head[0]['W_add'])+self.head[0]['b_add']
        erase = TT.nnet.sigmoid(TT.dot(h, self.head[0]['W_erase'])+self.head[0]['b_erase'])
        print key.ndim
        print beta.ndim
        print add.ndim
        print erase.ndim
        sim = cosine_sim(key, memory_before)
        weight_c = TT.nnet.softmax(beta.reshape((1,))*sim)
        weight_shift = TT.dot(weight_c, self.head[0]['W_sim'])+TT.dot(weight_before, self.head[0]['W_old'])+self.head[0]['b_old']
        weight_shift = TT.nnet.softmax(weight_shift)
        weight = weight_shift
        weight = weight.reshape((weight.shape[1],))
        weight_dim = weight.dimshuffle((0, 'x'))
        erase_dim = erase.reshape((erase.shape[0],)).dimshuffle(('x', 0))
        add_dim = add.reshape((add.shape[0],)).dimshuffle(('x', 0))
        memory_erase = memory_before*(1-(weight_dim*erase_dim))
        memory = memory_erase+(weight_dim*add_dim)
        return weight, memory

    def write_neuralhead_process_batch(self,h,weight_before,memory_before):
        key = TT.dot(h, self.head[0]['W_key'])+self.head[0]['b_key']
        beta = TT.nnet.softplus(TT.dot(h, self.head[0]['W_beta'])+self.head[0]['b_beta'])
        add = TT.dot(h, self.head[0]['W_add'])+self.head[0]['b_add']
        erase = TT.nnet.sigmoid(TT.dot(h, self.head[0]['W_erase'])+self.head[0]['b_erase'])
        print key.ndim
        print beta.ndim
        print add.ndim
        print erase.ndim
        sim = cosine_sim_batch(key,memory_before)
        weight_c = TT.nnet.softmax(beta.reshape((beta.shape[0],)).dimshuffle(0,'x')*sim)
        weight_shift = TT.dot(weight_c, self.head[0]['W_sim'])+TT.dot(weight_before, self.head[0]['W_old'])+self.head[0]['b_old']
        weight_shift = TT.nnet.softmax(weight_shift)
        weight = weight_shift
        weight_dim = weight.dimshuffle((0, 1,'x'))
        erase_dim = erase.dimshuffle((0,'x', 1))
        add_dim = add.dimshuffle((0,'x', 1))
        memory_erase = memory_before*(1-(weight_dim*erase_dim))
        memory = memory_erase+(weight_dim*add_dim)
        return weight, memory

    def init_neuralhead(self, num):

        logger.debug('neural head num: %d', num)
        self.head = [{}]
        for i in range(num):
            self.head[i]['W_input'] = theano.shared(
                                        self.memory_init_fn(self.memory_dim,
                                            self.n_hids,
                                            self.sparsity,
                                            self.scale,
                                            self.rng), 
                                            name='W_head_input_%s'%self.name)
            self.head[i]['W_reset'] = theano.shared(
                                        self.memory_init_fn(self.memory_dim,
                                            self.n_hids,
                                            self.sparsity,
                                            self.scale,
                                            self.rng), 
                                            name='W_head_reset_%s'%self.name)
            self.head[i]['W_update'] = theano.shared(
                                        self.memory_init_fn(self.memory_dim,
                                            self.n_hids,
                                            self.sparsity,
                                            self.scale,
                                            self.rng), 
                                            name='W_head_update_%s'%self.name)
            self.params.append(self.head[i]['W_input'])
            self.params.append(self.head[i]['W_reset'])
            self.params.append(self.head[i]['W_update'])
            self.head[i]['W_readkey'] = theano.shared(
                                        self.memory_init_fn(self.n_hids,
                                            self.memory_dim,
                                            self.sparsity,
                                            self.scale,
                                            self.rng), 
                                            name='W_head_readkey_%s'%self.name)
            self.head[i]['b_readkey'] = theano.shared(
                                        self.bias_fn(self.memory_dim,self.bias_scale,rng=self.rng),
                                        name="b_head_readkey_%s"%self.name)
            self.head[i]['W_readbeta'] = theano.shared(
                                        self.memory_init_fn(self.n_hids,
                                            1,
                                            self.sparsity,
                                            self.scale,
                                            self.rng), 
                                            name='W_head_readbeta_%s'%self.name)
            self.head[i]['b_readbeta'] = theano.shared(
                                        self.bias_fn(1,self.bias_scale,rng=self.rng),
                                        name="b_head_readbeta_%s"%self.name)
            self.head[i]['W_readold'] = theano.shared(
                                        self.init_fn(self.memory_size,
                                            self.memory_size,
                                            self.sparsity,
                                            self.scale,
                                            self.rng), 
                                            name='W_head_readold_%s'%self.name)
            self.head[i]['b_readold'] = theano.shared(
                                        self.bias_fn(self.memory_size,self.bias_scale,rng=self.rng),
                                        name="b_head_readold_%s"%self.name)
            self.head[i]['W_readsim'] = theano.shared(
                                        self.init_fn(self.memory_size,
                                            self.memory_size,
                                            self.sparsity,
                                            self.scale,
                                            self.rng), 
                                            name='W_head_readsim_%s'%self.name)
            self.params.append(self.head[i]['W_readkey'])
            self.params.append(self.head[i]['b_readkey'])
            self.params.append(self.head[i]['W_readbeta'])
            self.params.append(self.head[i]['b_readbeta'])
            self.params.append(self.head[i]['W_readsim'])
            self.params.append(self.head[i]['W_readold'])
            self.params.append(self.head[i]['b_readold'])
            self.head[i]['W_key'] = theano.shared(
                                        self.memory_init_fn(self.n_hids,
                                            self.memory_dim,
                                            self.sparsity,
                                            self.scale,
                                            self.rng), 
                                            name='W_head_key_%s'%self.name)
            self.head[i]['b_key'] = theano.shared(
                                        self.bias_fn(self.memory_dim,self.bias_scale,rng=self.rng),
                                        name="b_head_key_%s"%self.name)
            self.head[i]['W_beta'] = theano.shared(
                                        self.memory_init_fn(self.n_hids,
                                            1,
                                            self.sparsity,
                                            self.scale,
                                            self.rng), 
                                            name='W_head_beta_%s'%self.name)
            self.head[i]['b_beta'] = theano.shared(
                                        self.bias_fn(1,self.bias_scale,rng=self.rng),
                                        name="b_head_beta_%s"%self.name)
            self.head[i]['W_old'] = theano.shared(
                                        self.init_fn(self.memory_size,
                                            self.memory_size,
                                            self.sparsity,
                                            self.scale,
                                            self.rng), 
                                            name='W_head_old_%s'%self.name)
            self.head[i]['b_old'] = theano.shared(
                                        self.bias_fn(self.memory_size,self.bias_scale,rng=self.rng),
                                        name="b_head_old_%s"%self.name)
            self.head[i]['W_sim'] = theano.shared(
                                        self.init_fn(self.memory_size,
                                            self.memory_size,
                                            self.sparsity,
                                            self.scale,
                                            self.rng), 
                                            name='W_head_sim_%s'%self.name)
            self.head[i]['W_erase'] = theano.shared(
                                        self.memory_init_fn(self.n_hids,
                                            self.memory_dim,
                                            self.sparsity,
                                            self.scale,
                                            self.rng), 
                                            name='W_head_erase_%s'%self.name)
            self.head[i]['b_erase'] = theano.shared(
                                        self.bias_fn(self.memory_dim,self.bias_scale,rng=self.rng),
                                        name="b_head_erase_%s"%self.name)
            self.head[i]['W_add'] = theano.shared(
                                        self.memory_init_fn(self.n_hids,
                                            self.memory_dim,
                                            self.sparsity,
                                            self.scale,
                                            self.rng), 
                                            name='W_head_add_%s'%self.name)
            self.head[i]['b_add'] = theano.shared(
                                        self.bias_fn(self.memory_dim,self.bias_scale,rng=self.rng),
                                        name="b_head_add_%s"%self.name)
            
            self.params.append(self.head[i]['W_key'])
            self.params.append(self.head[i]['b_key'])
            self.params.append(self.head[i]['W_beta'])
            self.params.append(self.head[i]['b_beta'])
            

            self.params.append(self.head[i]['W_sim'])
            self.params.append(self.head[i]['W_old'])
            self.params.append(self.head[i]['b_old'])

            self.params.append(self.head[i]['W_erase'])
            self.params.append(self.head[i]['b_erase'])
            self.params.append(self.head[i]['W_add'])
            self.params.append(self.head[i]['b_add'])   
        self.read_head_process = self.read_neuralhead_process
        self.read_head_process_batch = self.read_neuralhead_process_batch     
        self.write_head_process = self.write_neuralhead_process
        self.write_head_process_batch = self.write_neuralhead_process_batch

    def read_attentionhead_process(self,h,inp,weight_before,memory_before):
        m_vec = TT.dot(memory_before, self.head[0]['W_readAM'])
        i_vec = TT.dot(inp, self.head[0]['W_readAI'])
        s_vec = TT.dot(h,self.head[0]['W_readAS'])
        inner = m_vec+s_vec+i_vec
        innert = TT.tanh(inner)
        ot = TT.dot(innert, self.head[0]['W_readAA'])
        otexp = TT.exp(ot).reshape((ot.shape[0],))
        normalizer = otexp.sum(axis=0)
        weight = otexp/normalizer
        return weight

    def read_attentionhead_process_batch(self,h,inp,weight_before,memory_before):
        m_vec = TT.dot(memory_before, self.head[0]['W_readAM'])
        i_vec = TT.dot(inp, self.head[0]['W_readAI']).dimshuffle(0,'x',1)
        s_vec = TT.dot(h,self.head[0]['W_readAS']).dimshuffle(0,'x',1)
        inner = m_vec+s_vec+i_vec
        innert = TT.tanh(inner)
        ot = TT.dot(innert, self.head[0]['W_readAA'])
        otexp = TT.exp(ot).reshape((ot.shape[0],ot.shape[1]))
        normalizer = otexp.sum(axis=1).dimshuffle(0,'x')
        weight = otexp/normalizer
        return weight

    def write_attentionhead_process(self,h,weight_before,memory_before):
        add = TT.dot(h, self.head[0]['W_add'])+self.head[0]['b_add']
        erase = TT.nnet.sigmoid(TT.dot(h, self.head[0]['W_erase'])+self.head[0]['b_erase'])
        m_vec = TT.dot(memory_before, self.head[0]['W_AM'])
        s_vec = TT.dot(h,self.head[0]['W_AS'])
        inner = m_vec+s_vec
        innert = TT.tanh(inner)
        ot = TT.dot(innert, self.head[0]['W_AA'])
        otexp = TT.exp(ot).reshape((ot.shape[0],))
        normalizer = otexp.sum(axis=0)
        weight = otexp/normalizer
        weight_dim = weight.dimshuffle((0, 'x'))
        erase_dim = erase.reshape((erase.shape[0],)).dimshuffle(('x', 0))
        add_dim = add.reshape((add.shape[0],)).dimshuffle(('x', 0))
        memory_erase = memory_before*(1-(weight_dim*erase_dim))
        memory = memory_erase+(weight_dim*add_dim)
        return weight, memory

    def write_attentionhead_process_batch(self,h,weight_before,memory_before):
        add = TT.dot(h, self.head[0]['W_add'])+self.head[0]['b_add']
        erase = TT.nnet.sigmoid(TT.dot(h, self.head[0]['W_erase'])+self.head[0]['b_erase'])
        m_vec = TT.dot(memory_before, self.head[0]['W_readAM'])
        s_vec = TT.dot(h,self.head[0]['W_readAS']).dimshuffle(0,'x',1)
        inner = m_vec+s_vec
        innert = TT.tanh(inner)
        ot = TT.dot(innert, self.head[0]['W_readAA'])
        otexp = TT.exp(ot).reshape((ot.shape[0],ot.shape[1]))
        normalizer = otexp.sum(axis=1).dimshuffle(0,'x')
        weight = otexp/normalizer
        weight_dim = weight.dimshuffle((0, 1,'x'))
        erase_dim = erase.dimshuffle((0,'x', 1))
        add_dim = add.dimshuffle((0,'x', 1))
        memory_erase = memory_before*(1-(weight_dim*erase_dim))
        memory = memory_erase+(weight_dim*add_dim)
        return weight, memory
    
    def init_attentionhead(self, num):

        logger.debug('attention head num: %d', num)
        self.head = [{}]
        for i in range(num):
            self.head[i]['W_input'] = theano.shared(
                                        self.memory_init_fn(self.memory_dim,
                                            self.n_hids,
                                            self.sparsity,
                                            self.scale,
                                            self.rng), 
                                            name='W_head_input_%s'%self.name)
            self.head[i]['W_reset'] = theano.shared(
                                        self.memory_init_fn(self.memory_dim,
                                            self.n_hids,
                                            self.sparsity,
                                            self.scale,
                                            self.rng), 
                                            name='W_head_reset_%s'%self.name)
            self.head[i]['W_update'] = theano.shared(
                                        self.memory_init_fn(self.memory_dim,
                                            self.n_hids,
                                            self.sparsity,
                                            self.scale,
                                            self.rng), 
                                            name='W_head_update_%s'%self.name)
            self.params.append(self.head[i]['W_input'])
            self.params.append(self.head[i]['W_reset'])
            self.params.append(self.head[i]['W_update'])
            self.head[i]['W_readAM'] = theano.shared(
                                        self.memory_init_fn(self.memory_dim,
                                            self.memory_dim,
                                            -1,
                                            10 ** (-3),
                                            self.rng), 
                                            name='W_head_readAM_%s'%self.name)
            self.head[i]['W_readAS'] = theano.shared(
                                        self.memory_init_fn(self.n_hids,
                                            self.memory_dim,
                                            -1,
                                            10 ** (-3),
                                            self.rng), 
                                            name='W_head_readAS_%s'%self.name)
            self.head[i]['W_readAI'] = theano.shared(
                                        self.memory_init_fn(self.n_hids,
                                            self.memory_dim,
                                            -1,
                                            10 ** (-3),
                                            self.rng), 
                                            name='W_head_readAS_%s'%self.name)
            self.head[i]['W_readAA'] = theano.shared(
                                        numpy.zeros((self.memory_dim, 1), dtype="float32"),
                                            name='W_head_readAA_%s'%self.name)
            self.params.append(self.head[i]['W_readAM'])
            self.params.append(self.head[i]['W_readAI'])
            self.params.append(self.head[i]['W_readAS'])
            self.params.append(self.head[i]['W_readAA'])
            self.head[i]['W_AM'] = theano.shared(
                                        self.memory_init_fn(self.memory_dim,
                                            self.memory_dim,
                                            -1,
                                            10 ** (-3),
                                            self.rng), 
                                            name='W_head_AM_%s'%self.name)
            self.head[i]['W_AS'] = theano.shared(
                                        self.memory_init_fn(self.n_hids,
                                            self.memory_dim,
                                            -1,
                                            10 ** (-3),
                                            self.rng), 
                                            name='W_head_AS_%s'%self.name)
            self.head[i]['W_AA'] = theano.shared(
                                        numpy.zeros((self.memory_dim, 1), dtype="float32"),
                                            name='W_head_AA_%s'%self.name)
            self.params.append(self.head[i]['W_AM'])
            self.params.append(self.head[i]['W_AS'])
            self.params.append(self.head[i]['W_AA'])
            self.head[i]['W_erase'] = theano.shared(
                                        self.memory_init_fn(self.n_hids,
                                            self.memory_dim,
                                            self.sparsity,
                                            self.scale,
                                            self.rng), 
                                            name='W_head_erase_%s'%self.name)
            self.head[i]['b_erase'] = theano.shared(
                                        self.bias_fn(self.memory_dim,self.bias_scale,rng=self.rng),
                                        name="b_head_erase_%s"%self.name)
            self.head[i]['W_add'] = theano.shared(
                                        self.memory_init_fn(self.n_hids,
                                            self.memory_dim,
                                            self.sparsity,
                                            self.scale,
                                            self.rng), 
                                            name='W_head_add_%s'%self.name)
            self.head[i]['b_add'] = theano.shared(
                                        self.bias_fn(self.memory_dim,self.bias_scale,rng=self.rng),
                                        name="b_head_add_%s"%self.name)
            self.params.append(self.head[i]['W_erase'])
            self.params.append(self.head[i]['b_erase'])
            self.params.append(self.head[i]['W_add'])
            self.params.append(self.head[i]['b_add'])   
        self.read_head_process = self.read_attentionhead_process
        self.read_head_process_batch = self.read_attentionhead_process_batch     
        self.write_head_process = self.write_attentionhead_process
        self.write_head_process_batch = self.write_attentionhead_process_batch

class NTMLayer(NTMLayerBase):
    """
        Standard recurrent layer with gates.
        See arXiv verion of our paper.
    """
    def __init__(self, rng,
                 n_hids=500,
                 scale=.01,
                 sparsity = -1,
                 activation = TT.tanh,
                 activ_noise=0.,
                 weight_noise=False,
                 bias_fn='init_bias',
                 bias_scale = 0.,
                 dropout = 1.,
                 init_fn='sample_weights',
                 kind_reg = None,
                 grad_scale = 1.,
                 profile = 0,
                 gating = False,
                 reseting = False,
                 gater_activation = TT.nnet.sigmoid,
                 reseter_activation = TT.nnet.sigmoid,
                 name=None,
                 rank_n_approx=100,
                 other_init_fn = 'sample_weights_classic',
                 memory_init_fn = 'sample_weights_classic',
                 init_memory_weight = True,
                 use_memory = True,
                 memory_size = 128,
                 memory_dim = 20,
                 head_fn = 'self.init_attentionhead',
                 head_num = 1,
                 memory_activation = TT.tanh):
        """
        :type rng: numpy random generator
        :param rng: numpy random generator

        :type n_in: int
        :param n_in: number of inputs units

        :type n_hids: int
        :param n_hids: Number of hidden units on each layer of the MLP

        :type activation: string/function or list of
        :param activation: Activation function for the embedding layers. If
            a list it needs to have a value for each layer. If not, the same
            activation will be applied to all layers

        :type scale: float or list of
        :param scale: depending on the initialization function, it can be
            the standard deviation of the Gaussian from which the weights
            are sampled or the largest singular value. If a single value it
            will be used for each layer, otherwise it has to have one value
            for each layer

        :type sparsity: int or list of
        :param sparsity: if a single value, it will be used for each layer,
            otherwise it has to be a list with as many values as layers. If
            negative, it means the weight matrix is dense. Otherwise it
            means this many randomly selected input units are connected to
            an output unit


        :type weight_noise: bool
        :param weight_noise: If true, the model is used with weight noise
            (and the right shared variable are constructed, to keep track of the
            noise)

        :type dropout: float
        :param dropout: the probability with which hidden units are dropped
            from the hidden layer. If set to 1, dropout is not used

        :type init_fn: string or function
        :param init_fn: function used to initialize the weights of the
            layer. We recommend using either `sample_weights_classic` or
            `sample_weights` defined in the utils

        :type bias_fn: string or function
        :param bias_fn: function used to initialize the biases. We recommend
            using `init_bias` defined in the utils

        :type bias_scale: float
        :param bias_scale: argument passed to `bias_fn`, depicting the scale
            of the initial bias

        :type grad_scale: float or theano scalar
        :param grad_scale: factor with which the gradients with respect to
            the parameters of this layer are scaled. It is used for
            differentiating between the different parameters of a model.

        :type gating: bool
        :param gating: If true, an update gate is used

        :type reseting: bool
        :param reseting: If true, a reset gate is used

        :type gater_activation: string or function
        :param name: The activation function of the update gate

        :type reseter_activation: string or function
        :param name: The activation function of the reset gate

        :type name: string
        :param name: name of the layer (used to name parameters). NB: in
            this library names are very important because certain parts of the
            code relies on name to disambiguate between variables, therefore
            each layer should have a unique name.

        """
        logger.debug("NTMLayer is used")
        self.grad_scale = grad_scale

        if type(init_fn) is str or type(init_fn) is unicode:
            init_fn = eval(init_fn)
        if type(bias_fn) is str or type(bias_fn) is unicode:
            bias_fn = eval(bias_fn)
        if type(activation) is str or type(activation) is unicode:
            activation = eval(activation)
        if type(gater_activation) is str or type(gater_activation) is unicode:
            gater_activation = eval(gater_activation)
        if type(reseter_activation) is str or type(reseter_activation) is unicode:
            reseter_activation = eval(reseter_activation)
        if type(other_init_fn) is str or type(other_init_fn) is unicode:
            other_init_fn = eval(other_init_fn)
        if type(memory_activation) is str or type(memory_activation) is unicode:
            memory_activation = eval(memory_activation)
        if type(memory_init_fn) is str or type(memory_init_fn) is unicode:
            memory_init_fn = eval(memory_init_fn)
        if type(head_fn) is str or type(head_fn) is unicode:
            head_fn = eval(head_fn)

        self.scale = scale
        self.sparsity = sparsity
        self.activation = activation
        self.n_hids = n_hids
        self.bias_scale = bias_scale
        self.bias_fn = bias_fn
        self.init_fn = init_fn
        self.weight_noise = weight_noise
        self.activ_noise = activ_noise
        self.profile = profile
        self.dropout = dropout
        self.gating = gating
        self.reseting = reseting
        self.gater_activation = gater_activation
        self.reseter_activation = reseter_activation
        self.rank_n_approx = rank_n_approx
        self.other_init_fn = other_init_fn
        self.memory_size = memory_size
        self.memory_dim = memory_dim
        self.init_memory_weight = init_memory_weight
        self.use_memory = use_memory
        self.head_fn = head_fn
        self.head_num = head_num
        self.memory_activation = memory_activation
        self.memory_init_fn = memory_init_fn

        assert rng is not None, "random number generator should not be empty!"

        super(NTMLayer, self).__init__(self.n_hids,
                self.n_hids, rng, name)

        self.trng = RandomStreams(self.rng.randint(int(1e6)))
        self.params = []
        self._init_params()

    def _init_params(self):
        self.W_hh = theano.shared(
                self.init_fn(self.n_hids,
                self.n_hids,
                self.sparsity,
                self.scale,
                rng=self.rng),
                name="W_%s"%self.name)
        self.params = [self.W_hh]
        if self.init_memory_weight and self.use_memory:
            logger.debug('memory & weight used')
            self.initial_memory = theano.shared(
                                    numpy.ones((self.memory_size, self.memory_dim), dtype="float32")/1e5,
                                        name='initial_memory_%s'%self.name)
            self.params.append(self.initial_memory)
            self.initial_read_weight = theano.shared(
                                    self.bias_fn(
                                        self.memory_size,
                                        1./self.memory_size,
                                        self.rng), 
                                        name='initial_read_weight_%s'%self.name)
            self.initial_write_weight = theano.shared(
                                    self.bias_fn(
                                        self.memory_size,
                                        1./self.memory_size,
                                        self.rng), 
                                        name='initial_write_weight_%s'%self.name)
            self.params.append(self.initial_read_weight)
            self.params.append(self.initial_write_weight)
        if self.use_memory:
            self.head_fn(self.head_num)
        if self.gating:
            self.G_hh = theano.shared(
                    self.init_fn(self.n_hids,
                    self.n_hids,
                    self.sparsity,
                    self.scale,
                    rng=self.rng),
                    name="G_%s"%self.name)
            self.params.append(self.G_hh)
        if self.reseting:
            self.R_hh = theano.shared(
                    self.init_fn(self.n_hids,
                    self.n_hids,
                    self.sparsity,
                    self.scale,
                    rng=self.rng),
                    name="R_%s"%self.name)
            self.params.append(self.R_hh)
        self.params_grad_scale = [self.grad_scale for x in self.params]
        self.restricted_params = [x for x in self.params]
        if self.weight_noise:
            self.nW_hh = theano.shared(self.W_hh.get_value()*0, name='noise_'+self.W_hh.name)
            self.noise_params = [self.nW_hh]
            if self.gating:
                self.nG_hh = theano.shared(self.G_hh.get_value()*0, name='noise_'+self.G_hh.name)
                self.noise_params += [self.nG_hh]
            if self.reseting:
                self.nR_hh = theano.shared(self.R_hh.get_value()*0, name='noise_'+self.R_hh.name)
                self.noise_params += [self.nR_hh]
            self.noise_params_shape_fn = [constant_shape(x.get_value().shape)
                            for x in self.noise_params]

    def step_fprop(self,
                   state_below,
                   mask = None,
                   state_before = None,
                   gater_below = None,
                   reseter_below = None,
                   memory_before = None,
                   read_weight_before = None,
                   write_weight_before = None,
                   use_noise=True,
                   no_noise_bias = False):
        """
        Constructs the computational graph of this layer.

        :type state_below: theano variable
        :param state_below: the input to the layer

        :type mask: None or theano variable
        :param mask: mask describing the length of each sequence in a
            minibatch

        :type state_before: theano variable
        :param state_before: the previous value of the hidden state of the
            layer

        :type gater_below: theano variable
        :param gater_below: the input to the update gate

        :type reseter_below: theano variable
        :param reseter_below: the input to the reset gate

        :type use_noise: bool
        :param use_noise: flag saying if weight noise should be used in
            computing the output of this layer

        :type no_noise_bias: bool
        :param no_noise_bias: flag saying if weight noise should be added to
            the bias as well
        """

        rval = []
        if self.weight_noise and use_noise and self.noise_params:
            W_hh = self.W_hh + self.nW_hh
            if self.gating:
                G_hh = self.G_hh + self.nG_hh
            if self.reseting:
                R_hh = self.R_hh + self.nR_hh
        else:
            W_hh = self.W_hh
            if self.gating:
                G_hh = self.G_hh
            if self.reseting:
                R_hh = self.R_hh

        #read from memory
        if read_weight_before.ndim == 2:
            read_weight_new = self.read_head_process_batch(state_before,state_below,read_weight_before,memory_before)
            read_weight_new_dim = read_weight_new.dimshuffle(0,1,'x')
            read_below = TT.sum(read_weight_new_dim*memory_before,axis=1)
        else:
            read_weight_new = self.read_head_process(state_before,state_below,read_weight_before,memory_before)
            read_below = TT.dot(read_weight_new, memory_before)

        state_below += TT.dot(read_below, self.head[0]['W_input'])
        reseter_below += TT.dot(read_below, self.head[0]['W_reset'])
        gater_below += TT.dot(read_below, self.head[0]['W_update'])

        # Reset gate:
        # optionally reset the hidden state.
        if self.reseting and reseter_below:
            reseter = self.reseter_activation(TT.dot(state_before, R_hh) +
                    reseter_below)
            reseted_state_before = reseter * state_before
        else:
            reseted_state_before = state_before

        # Feed the input to obtain potential new state.
        preactiv = TT.dot(reseted_state_before, W_hh) + state_below
        h = self.activation(preactiv)

        # Update gate:
        # optionally reject the potential new state and use the new one.
        if self.gating and gater_below:
            gater = self.gater_activation(TT.dot(state_before, G_hh) +
                    gater_below)
            h = gater * h + (1-gater) * state_before

        #update the weights and memories
        if write_weight_before.ndim == 2:
            write_weight_new, memory_new = self.write_head_process_batch(h,write_weight_before,memory_before)
        else:
            write_weight_new, memory_new = self.write_head_process(h,write_weight_before,memory_before)

        if self.activ_noise and use_noise:
            h = h + self.trng.normal(h.shape, avg=0, std=self.activ_noise, dtype=h.dtype)
        if mask is not None:
            if h.ndim ==2 and mask.ndim==1:
                mask = mask.dimshuffle(0,'x')
            h = mask * h + (1-mask) * state_before
        if self.use_memory:
            return h, memory_new, read_weight_new,write_weight_new
        return h

    def fprop(self,
              state_below,
              mask=None,
              init_state=None,
              init_memory=None,
              init_read_weight=None,
              init_write_weight=None,
              gater_below=None,
              reseter_below=None,
              nsteps=None,
              batch_size=None,
              use_noise=True,
              truncate_gradient=-1,
              no_noise_bias = False
             ):

        if theano.config.floatX=='float32':
            floatX = numpy.float32
        else:
            floatX = numpy.float64
        if nsteps is None:
            nsteps = state_below.shape[0]
            if batch_size and batch_size != 1:
                nsteps = nsteps / batch_size
        if batch_size is None and state_below.ndim == 3:
            batch_size = state_below.shape[1]
        if state_below.ndim == 2 and \
           (not isinstance(batch_size,int) or batch_size > 1):
            state_below = state_below.reshape((nsteps, batch_size, self.n_in))
            if gater_below:
                gater_below = gater_below.reshape((nsteps, batch_size, self.n_in))
            if reseter_below:
                reseter_below = reseter_below.reshape((nsteps, batch_size, self.n_in))

        if not init_state:
            if not isinstance(batch_size, int) or batch_size != 1:
                init_state = TT.alloc(floatX(0), batch_size, self.n_hids)
            else:
                init_state = TT.alloc(floatX(0), self.n_hids)

        #assume that gate and reset is used
        assert self.gating and gater_below
        assert self.reseting and reseter_below
        assert self.use_memory
        if not init_memory:
            if not isinstance(batch_size, int) or batch_size != 1:
                init_memory = TT.alloc(self.initial_memory, batch_size, *self.initial_memory.shape)
            else:
                init_memory = self.initial_memory
        if not init_read_weight:
            if not isinstance(batch_size, int) or batch_size != 1:
                init_read_weight = TT.alloc(self.initial_read_weight, batch_size, *self.initial_read_weight.shape)
            else:
                init_read_weight = self.initial_read_weight
        if not init_write_weight:
            if not isinstance(batch_size, int) or batch_size != 1:
                init_write_weight = TT.alloc(self.initial_write_weight, batch_size, *self.initial_write_weight.shape)
            else:
                init_write_weight = self.initial_write_weight
        outps = [init_state, init_memory, init_read_weight, init_write_weight]
        if mask:
            inps = [state_below, mask, gater_below, reseter_below]
            fn = lambda x,y,g,r,z,m,rw, ww : self.step_fprop(x,y,z,
                                                    gater_below=g,
                                                    reseter_below=r,
                                                    memory_before=m,
                                                    read_weight_before=rw,
                                                    write_weight_before=ww,
                                                    use_noise=use_noise,
                                                    no_noise_bias=no_noise_bias)
        else:
            inps = [state_below, gater_below, reseter_below]
            fn = lambda tx, ty,tg,tr,tm,trw,tww: self.step_fprop(tx, None, ty,
                                                    gater_below=tg,
                                                    reseter_below=tr,
                                                    memory_before=tm,
                                                    read_weight_before=trw,
                                                    write_weight_before=tww,
                                                    use_noise=use_noise,
                                                    no_noise_bias=no_noise_bias)
        '''
        else:
            outps = [init_state]
            if mask:
                inps = [state_below, mask]
                fn = lambda x,y,z : self.step_fprop(x,y,z, use_noise=use_noise,
                                                    no_noise_bias=no_noise_bias)
            else:
                inps = [state_below]
                fn = lambda tx, ty: self.step_fprop(tx, None, ty,
                                                    use_noise=use_noise,
                                                    no_noise_bias=no_noise_bias)
        '''

        #self.fun = init_memory

        rvalss, updates = theano.scan(fn,
                        sequences = inps,
                        outputs_info = outps,
                        name='layer_%s'%self.name,
                        profile=self.profile,
                        truncate_gradient = truncate_gradient,
                        n_steps = nsteps)
        self.rvalss = rvalss
        rval = rvalss[0:4]
        new_h = rval
        self.out = rval
        self.rval = rval
        self.updates =updates

        return self.out

class NTMLayerWithSearch(NTMLayerBase):
    """A copy of RecurrentLayer from groundhog"""

    def __init__(self, rng,
                 n_hids,
                 c_dim=None,
                 scale=.01,
                 activation=TT.tanh,
                 bias_fn='init_bias',
                 bias_scale=0.,
                 init_fn='sample_weights',
                 gating=False,
                 reseting=False,
                 dropout=1.,
                 gater_activation=TT.nnet.sigmoid,
                 reseter_activation=TT.nnet.sigmoid,
                 weight_noise=False,
                 name=None,
                 rank_n_approx=100,
                 sparsity = -1,
                 other_init_fn = 'sample_weights_classic',
                 memory_init_fn = 'sample_weights_classic',
                 init_memory_weight = True,
                 use_memory = True,
                 memory_size = 128,
                 memory_dim = 20,
                 head_num = 1,
                 memory_activation = TT.tanh):
        logger.debug("NTMLayerWithSearch is used")

        self.grad_scale = 1
        assert gating == True
        assert reseting == True
        assert dropout == 1.
        assert weight_noise == False
        updater_activation = gater_activation

        if type(init_fn) is str or type(init_fn) is unicode:
            init_fn = eval(init_fn)
        if type(bias_fn) is str or type(bias_fn) is unicode:
            bias_fn = eval(bias_fn)
        if type(activation) is str or type(activation) is unicode:
            activation = eval(activation)
        if type(updater_activation) is str or type(updater_activation) is unicode:
            updater_activation = eval(updater_activation)
        if type(reseter_activation) is str or type(reseter_activation) is unicode:
            reseter_activation = eval(reseter_activation)
        if type(other_init_fn) is str or type(other_init_fn) is unicode:
            other_init_fn = eval(other_init_fn)
        if type(memory_activation) is str or type(memory_activation) is unicode:
            memory_activation = eval(memory_activation)
        if type(memory_init_fn) is str or type(memory_init_fn) is unicode:
            memory_init_fn = eval(memory_init_fn)

        self.scale = scale
        self.activation = activation
        self.n_hids = n_hids
        self.bias_scale = bias_scale
        self.bias_fn = bias_fn
        self.init_fn = init_fn
        self.updater_activation = updater_activation
        self.reseter_activation = reseter_activation
        self.c_dim = c_dim
        self.gating = gating
        self.reseting = reseting
        self.rank_n_approx = rank_n_approx
        self.sparsity = sparsity
        self.other_init_fn = other_init_fn
        self.memory_size = memory_size
        self.memory_dim = memory_dim
        self.init_memory_weight = init_memory_weight
        self.use_memory = use_memory
        self.head_num = head_num
        self.memory_activation = memory_activation
        self.memory_init_fn = memory_init_fn

        assert rng is not None, "random number generator should not be empty!"

        super(NTMLayerWithSearch, self).__init__(self.n_hids,
                self.n_hids, rng, name)

        self.params = []
        self._init_params()

    def _init_params(self):
        self.W_hh = theano.shared(
                self.init_fn(self.n_hids,
                self.n_hids,
                -1,
                self.scale,
                rng=self.rng),
                name="W_%s"%self.name)
        self.params = [self.W_hh]
        self.G_hh = theano.shared(
                self.init_fn(self.n_hids,
                    self.n_hids,
                    -1,
                    self.scale,
                    rng=self.rng),
                name="G_%s"%self.name)
        self.params.append(self.G_hh)
        self.R_hh = theano.shared(
                self.init_fn(self.n_hids,
                    self.n_hids,
                    -1,
                    self.scale,
                    rng=self.rng),
                name="R_%s"%self.name)
        self.params.append(self.R_hh)
        self.A_cp = theano.shared(
                sample_weights_classic(self.c_dim,
                    self.n_hids,
                    -1,
                    10 ** (-3),
                    rng=self.rng),
                name="A_%s"%self.name)
        self.params.append(self.A_cp)
        self.B_hp = theano.shared(
                sample_weights_classic(self.n_hids,
                    self.n_hids,
                    -1,
                    10 ** (-3),
                    rng=self.rng),
                name="B_%s"%self.name)
        self.params.append(self.B_hp)
        self.D_pe = theano.shared(
                numpy.zeros((self.n_hids, 1), dtype="float32"),
                name="D_%s"%self.name)
        self.params.append(self.D_pe)
        '''
        if self.init_memory_weight and self.use_memory:
            logger.debug('memory & weight used')
            self.initial_memory = theano.shared(
                                    self.memory_init_fn(self.memory_size,
                                        self.memory_dim,
                                        self.sparsity,
                                        self.scale,
                                        self.rng), 
                                        name='initial_memory_%s'%self.name)
            self.params.append(self.initial_memory)
            self.initial_weight = theano.shared(
                                    self.bias_fn(
                                        self.memory_size,
                                        1./self.memory_size,
                                        self.rng), 
                                        name='initial_weight_%s'%self.name)
            self.params.append(self.initial_weight)

        if self.use_memory:
            self.init_head(1)
        '''
        self.params_grad_scale = [self.grad_scale for x in self.params]
       
    def set_decoding_layers(self, c_inputer, c_reseter, c_updater):
        self.c_inputer = c_inputer
        self.c_reseter = c_reseter
        self.c_updater = c_updater
        for layer in [c_inputer, c_reseter, c_updater]:
            self.params += layer.params
            self.params_grad_scale += layer.params_grad_scale

    def step_fprop(self,
                   state_below,
                   state_before,
                   gater_below=None,
                   reseter_below=None,
                   weight_before=None,
                   memory_before=None,
                   mask=None,
                   c=None,
                   c_mask=None,
                   p_from_c=None,
                   use_noise=True,
                   no_noise_bias=False,
                   step_num=None,
                   return_alignment=False):
        """
        Constructs the computational graph of this layer.

        :type state_below: theano variable
        :param state_below: the input to the layer

        :type mask: None or theano variable
        :param mask: mask describing the length of each sequence in a
            minibatch

        :type state_before: theano variable
        :param state_before: the previous value of the hidden state of the
            layer

        :type updater_below: theano variable
        :param updater_below: the input to the update gate

        :type reseter_below: theano variable
        :param reseter_below: the input to the reset gate

        :type use_noise: bool
        :param use_noise: flag saying if weight noise should be used in
            computing the output of this layer

        :type no_noise_bias: bool
        :param no_noise_bias: flag saying if weight noise should be added to
            the bias as well
        """

        updater_below = gater_below

        W_hh = self.W_hh
        G_hh = self.G_hh
        R_hh = self.R_hh
        A_cp = self.A_cp
        B_hp = self.B_hp
        D_pe = self.D_pe

        # The code works only with 3D tensors
        cndim = c.ndim
        if cndim == 2:
            c = c[:, None, :]

        # Warning: either source_num or target_num should be equal,
        #          or on of them sould be 1 (they have to broadcast)
        #          for the following code to make any sense.
        source_len = c.shape[0]
        source_num = c.shape[1]
        target_num = state_before.shape[0]
        dim = self.n_hids

        # Form projection to the tanh layer from the previous hidden state
        # Shape: (source_len, target_num, dim)
        p_from_h = ReplicateLayer(source_len)(utils.dot(state_before, B_hp)).out

        # Form projection to the tanh layer from the source annotation.
        if not p_from_c:
            p_from_c =  utils.dot(c, A_cp).reshape((source_len, source_num, dim))

        # Sum projections - broadcasting happens at the dimension 1.
        p = p_from_h + p_from_c

        # Apply non-linearity and project to energy.
        energy = TT.exp(utils.dot(TT.tanh(p), D_pe)).reshape((source_len, target_num))
        if c_mask:
            # This is used for batches only, that is target_num == source_num
            energy *= c_mask

        # Calculate energy sums.
        normalizer = energy.sum(axis=0)

        # Get probabilities.
        probs = energy / normalizer

        # Calculate weighted sums of source annotations.
        # If target_num == 1, c shoulds broadcasted at the 1st dimension.
        # Probabilities are broadcasted at the 2nd dimension.
        ctx = (c * probs.dimshuffle(0, 1, 'x')).sum(axis=0)

        state_below += self.c_inputer(ctx).out
        reseter_below += self.c_reseter(ctx).out
        updater_below += self.c_updater(ctx).out

        #read from memory
        '''
        if weight_before.ndim == 2:
            read_below = TT.batched_dot(weight_before, memory_before)
        else:
            read_below = TT.dot(weight_before, memory_before)

        state_below += TT.dot(read_below, self.head[0]['W_input'])
        reseter_below += TT.dot(read_below, self.head[0]['W_reset'])
        updater_below += TT.dot(read_below, self.head[0]['W_update'])
        '''

        # Reset gate:
        # optionally reset the hidden state.
        reseter = self.reseter_activation(TT.dot(state_before, R_hh) +
                reseter_below)
        reseted_state_before = reseter * state_before

        # Feed the input to obtain potential new state.
        preactiv = TT.dot(reseted_state_before, W_hh) + state_below
        h = self.activation(preactiv)

        # Update gate:
        # optionally reject the potential new state and use the new one.
        updater = self.updater_activation(TT.dot(state_before, G_hh) +
                updater_below)
        h = updater * h + (1-updater) * state_before

        #update the weights and memories
        '''
        key = TT.dot(h, self.head[0]['W_key'])+self.head[0]['b_key']
        beta = TT.nnet.softplus(TT.dot(h, self.head[0]['W_beta'])+self.head[0]['b_beta'])
        g = TT.nnet.sigmoid(TT.dot(h, self.head[0]['W_g'])+self.head[0]['b_g'])
        add = TT.dot(h, self.head[0]['W_add'])+self.head[0]['b_add']
        erase = TT.nnet.sigmoid(TT.dot(h, self.head[0]['W_erase'])+self.head[0]['b_erase'])
        if key.ndim == 2:
            [weight_new, memory_new],_ = theano.scan(
                                    self.head_process,
                                    sequences=[key,beta,g,add,erase,weight_before,memory_before],
                                    n_steps=key.shape[0])
        else:
            weight_new, memory_new = self.head_process(key,beta,g,add,erase,weight_before,memory_before)
        '''
        if mask is not None:
            if h.ndim ==2 and mask.ndim==1:
                mask = mask.dimshuffle(0,'x')
            h = mask * h + (1-mask) * state_before

        results = [h, ctx]
        if return_alignment:
            results += [probs]
        return results

    def fprop(self,
              state_below,
              mask=None,
              init_state=None,
              gater_below=None,
              reseter_below=None,
              c=None,
              c_mask=None,
              nsteps=None,
              batch_size=None,
              use_noise=True,
              truncate_gradient=-1,
              no_noise_bias=False,
              return_alignment=False):

        
        assert state_below
        assert gater_below and self.gating
        assert reseter_below and self.reseting
        assert self.use_memory
        

        updater_below = gater_below

        if theano.config.floatX=='float32':
            floatX = numpy.float32
        else:
            floatX = numpy.float64
        if nsteps is None:
            nsteps = state_below.shape[0]
            if batch_size and batch_size != 1:
                nsteps = nsteps / batch_size
        if batch_size is None and state_below.ndim == 3:
            batch_size = state_below.shape[1]
        if state_below.ndim == 2 and \
           (not isinstance(batch_size,int) or batch_size > 1):
            state_below = state_below.reshape((nsteps, batch_size, self.n_in))
            if updater_below:
                updater_below = updater_below.reshape((nsteps, batch_size, self.n_in))
            if reseter_below:
                reseter_below = reseter_below.reshape((nsteps, batch_size, self.n_in))

        assert init_state
        if not init_state:
            if not isinstance(batch_size, int) or batch_size != 1:
                init_state = TT.alloc(floatX(0), batch_size, self.n_hids)
            else:
                init_state = TT.alloc(floatX(0), self.n_hids)
        '''
        if not init_memory:
            print 'build init memory'
            if not isinstance(batch_size, int) or batch_size != 1:
                init_memory = TT.alloc(self.initial_memory, batch_size, *self.initial_memory.shape)
            else:
                init_memory = self.initial_memory
        if not init_weight:
            if not isinstance(batch_size, int) or batch_size != 1:
                init_weight = TT.alloc(self.initial_weight, batch_size, *self.initial_weight.shape)
            else:
                init_weight = self.initial_weight
        '''

        p_from_c =  utils.dot(c, self.A_cp).reshape(
                (c.shape[0], c.shape[1], self.n_hids))
        
        if mask:
            sequences = [state_below, mask, updater_below, reseter_below]
            non_sequences = [c, c_mask, p_from_c] 
            #              seqs    |     out    |  non_seqs
            fn = lambda x, m, g, r,   h, c1, cm, pc : self.step_fprop(x, h, mask=m,
                    gater_below=g, reseter_below=r,
                    #weight_before=w, memory_before=mem,
                    c=c1, p_from_c=pc, c_mask=cm,
                    use_noise=use_noise, no_noise_bias=no_noise_bias,
                    return_alignment=return_alignment)
        else:
            sequences = [state_below, updater_below, reseter_below]
            non_sequences = [c, p_from_c]
            #            seqs   |    out   | non_seqs
            fn = lambda x, g, r,   h,    c1, pc : self.step_fprop(x, h,
                    gater_below=g, reseter_below=r,
                    #weight_before=w, memory_before=mem,
                    c=c1, p_from_c=pc,
                    use_noise=use_noise, no_noise_bias=no_noise_bias,
                    return_alignment=return_alignment)

        outputs_info = [init_state,None]
        if return_alignment:
            outputs_info.append(None)

        rval, updates = theano.scan(fn,
                        sequences=sequences,
                        non_sequences=non_sequences,
                        outputs_info=outputs_info,
                        name='layer_%s'%self.name,
                        truncate_gradient=truncate_gradient,
                        n_steps=nsteps)
        self.out = rval
        self.rval = rval
        self.updates = updates

        return self.out
