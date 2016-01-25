#!/usr/bin/env python

import argparse
import cPickle
import logging
import pprint
import time

import numpy
import theano

from theano.printing import pydotprint
from groundhog.trainer.SGD_adadelta import SGD as SGD_adadelta
from groundhog.trainer.SGD import SGD as SGD
from groundhog.trainer.SGD_momentum import SGD as SGD_momentum
from groundhog.mainLoop import MainLoop
from experiments.nmt import\
        RNNEncoderDecoder, NTMEncoderDecoder, prototype_state, get_batch_iterator
import experiments.nmt

logger = logging.getLogger(__name__)

class RandomSamplePrinter(object):

    def __init__(self, state, model, train_iter):
        args = dict(locals())
        args.pop('self')
        self.__dict__.update(**args)

    def __call__(self):
        def cut_eol(words):
            for i, word in enumerate(words):
                if words[i] == '<eol>':
                    return words[:i + 1]
            raise Exception("No end-of-line found")

        sample_idx = 0
        while sample_idx < self.state['n_examples']:
            batch = self.train_iter.next(peek=True)
            xs, ys = batch['x'], batch['y']
            for seq_idx in range(xs.shape[1]):
                if sample_idx == self.state['n_examples']:
                    break

                x, y = xs[:, seq_idx], ys[:, seq_idx]
                x_words = cut_eol(map(lambda w_idx : self.model.word_indxs_src[w_idx], x))
                y_words = cut_eol(map(lambda w_idx : self.model.word_indxs[w_idx], y))
                if len(x_words) == 0:
                    continue

                print "Input: {}".format(" ".join(x_words))
                print "Target: {}".format(" ".join(y_words))
                self.model.get_samples(self.state['seqlen'] + 1, self.state['n_samples'], x[:len(x_words)])
                sample_idx += 1

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--state", help="State to use")
    parser.add_argument("--proto",  default="prototype_state",
        help="Prototype state to use for state")
    parser.add_argument("--skip-init", action="store_true",
        help="Skip parameter initilization")
    parser.add_argument("changes",  nargs="*", help="Changes to state", default="")
    return parser.parse_args()

def main():
    args = parse_args()

    state = getattr(experiments.nmt, args.proto)()

    if args.state:
        if args.state.endswith(".py"):
            state.update(eval(open(args.state).read()))
        else:
            with open(args.state) as src:
                state.update(cPickle.load(src))
    for change in args.changes:
        state.update(eval("dict({})".format(change)))

    logging.basicConfig(level=getattr(logging, state['level']), format="%(asctime)s: %(name)s: %(levelname)s: %(message)s")
    logger.debug("State:\n{}".format(pprint.pformat(state)))

    rng = numpy.random.RandomState(state['seed'])
    if args.proto == 'prototype_ntm_state':
        print 'Neural Turing Machine'
        enc_dec = NTMEncoderDecoder(state, rng, args.skip_init)
    else:
        enc_dec = RNNEncoderDecoder(state, rng, args.skip_init)
    enc_dec.build()
    lm_model = enc_dec.create_lm_model()

    
    logger.debug("Load data")
    train_data = get_batch_iterator(state)
    train_data.start(-1)
    logger.debug("Compile trainer")
    #algo = eval(state['algo'])(lm_model, state, train_data)

    #algo()
    #train
    print '---test training---'
    for i in range(1):
        batch = train_data.next()
        #print batch
        x = batch['x']
        print x.shape
        y = batch['y']
        print y.shape
        x_mask = batch['x_mask']
        y_mask = batch['y_mask']
        train_outputs = enc_dec.forward_training.rvalss+[
                    enc_dec.training_c.out,
                    enc_dec.forward_training_c.out,
                    enc_dec.forward_training_m.out,
                    enc_dec.forward_training_rw.out,
                    enc_dec.backward_training_c.out,
                    enc_dec.backward_training_m.out,
                    enc_dec.backward_training_rw.out,
                    
                    ]
        train_outputs = enc_dec.forward_training.rvalss+[enc_dec.predictions.out]
        test_train = theano.function(inputs=[enc_dec.x, enc_dec.x_mask, enc_dec.y, enc_dec.y_mask],
                                    outputs=train_outputs)
        result = test_train(x,x_mask,y,y_mask)
        for i in result:
            print i.shape
    #sample
    #batch = train_data.next()
    #print batch
    print '---test sampling---'
    x = [213,24242,542,144,30000]
    n_samples=10
    n_steps=10
    T=1
    inps = [enc_dec.sampling_x,
            enc_dec.n_samples,
            enc_dec.n_steps,
            enc_dec.T]
    #test_sample = theano.function(inputs=[enc_dec.sampling_x],
    #                            outputs=[enc_dec.sample])
    test_outputs = [enc_dec.sampling_c,
                    enc_dec.forward_sampling_c,
                    enc_dec.forward_sampling_m,
                    enc_dec.forward_sampling_rw,
                    enc_dec.backward_sampling_c,
                    enc_dec.backward_sampling_m,
                    enc_dec.backward_sampling_rw
                    ]
    test_outputs = enc_dec.forward_sampling.rvalss#+[enc_dec.sample,enc_dec.sample_log_prob,enc_dec.sampling_updates]
    #test_outputs = [enc_dec.sample,enc_dec.sample_log_prob]
    #sample_fn = theano.function(inputs=inps,outputs=test_outputs)
    sampler = enc_dec.create_sampler(many_samples=True)
    result = sampler(n_samples, n_steps,T,x)
    print result
    '''
    lm_model = enc_dec.create_lm_model()
    #pydotprint(enc_dec.predictions, outfile='~/Desktop/mtgraph.png', var_with_name_simple = True)

    logger.debug("Load data")
    train_data = get_batch_iterator(state)
    logger.debug("Compile trainer")
    algo = eval(state['algo'])(lm_model, state, train_data)
    logger.debug("Run training")
    main = MainLoop(train_data, None, None, lm_model, algo, state, None,
            reset=state['reset'],
            hooks=[RandomSamplePrinter(state, lm_model, train_data)]
                if state['hookFreq'] >= 0
                else None)
    if state['reload']:
        main.load()
    if state['loopIters'] > 0:
        main.main()
    '''

if __name__ == "__main__":
    main()
