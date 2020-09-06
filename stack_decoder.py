#!/usr/bin/env python
# -*- coding: utf-8 -*-

import math
from functools import total_ordering

from RedBlackTree.rbtree import RedBlackNode
from RedBlackTree.rbtree import RedBlackTree

import tensorflow as tf


def from_logit_to_log_softmax( logits ):

    '''converts logit to softmax in Tensor of tensorflow'''

    norm_logits = logits - tf.math.reduce_max( logits )

    exp_logits = tf.math.exp( norm_logits )

    exp_sum = tf.math.reduce_sum( exp_logits )

    ln_exp_sum = tf.math.log ( exp_sum )

    return  norm_logits - ln_exp_sum


@total_ordering
class StackDecoderPath:

    '''
    represents a explored path during a search

    It consists of log probability and sentence.
    The sentence is represented by a list of token indices.
    '''

    def __init__( self, log_prob, sentence, dec_state ):

        self.sentence  = sentence   # list of words in token index
        self.log_prob  = log_prob   # log probability for the sentence
        self.dec_state = dec_state  # decoder hidden state.

        # NOTE on dec_state:
        # This is an output from the decoder when sentence[-2] was processed.
        # This is given to the decoder together with sentence[-1] to predict
        # the next word.

    def last_word(self):
        return self.sentence[-1]

    def __eq__( self, other ):

        return self.log_prob == other.log_prob

    def __ne__( self, other ):

        return not (self == other)


    def __lt__( self, other ):
        return self.log_prob < other.log_prob


class StackDecoderQueue():

    '''
    represents a priority queue to hold the explored paths 
    in the order of their probability
    '''

    def __init__(self):
        self.rbtree   = RedBlackTree()

    def size(self):
        return self.rbtree.size()

    def pop_highest(self):
        if self.rbtree.size() > 0:
            n = self.rbtree.TREE_MAXIMUM( self.rbtree.ROOT() )
            self.rbtree.RB_DELETE(n)
            return n.val()
        else:
            return None

    def pop_lowest(self):
        if self.rbtree.size() > 0:
            n = self.rbtree.TREE_MINIMUM( self.rbtree.ROOT() )
            self.rbtree.RB_DELETE(n)
            return n.val()
        else:
            return None

    def min(self):
        ''' returns the log probability of the lowest path'''

        if self.rbtree.size() > 0:
            n = self.rbtree.TREE_MINIMUM( self.rbtree.ROOT() )
            return n.val().log_prob
        else:
            None

    def insert(self, path):
        n = RedBlackNode(path)

        self.rbtree.RB_INSERT(n) 


class StackDecoder():

    '''
    stack decoder for seq2seq models for tensorflow 2.x.
    
    description:
    Starting from BOS, it discovers the partial
    sentence of high probability by incrementally applying
    decoding until it reaches EOS.
    Please see 10.2 A*("Stack") Decoding in [JM09].

    purpose:
    Use this in order to generate top N-best sentences.
    Given the same encoder output and state, it always gives the
    same output, and the resultant sentences can be similar as in
    very low edit distance to each other.

    use tfa.seq2seq.BeamSearchDecoder if you want variety of
    outputs by sampling.

    reference:
    [JM09] Speech and Language Processing by D. Jurafsky & J.H. Martin2009 Pearson
    '''

    def __init__(self, decoder, BOS, EOS, use_attn = True):
        self.decoder   = decoder  # RNN decoder (LSTM, GRU etc)
        self.BOS       = BOS      # Beginnng of sentence token index (usually 1)
        self.EOS       = EOS      # End of sentence token index (usually 2)
        self.use_attn  = use_attn # True if an attenton mechanism is used i.e.,
                                  # the entire encoder state is needed for decoding.
        self.num_evals = 0


    def NBest(self, enc_out, enc_state, beam_width, num_nbest, max_len_sentence ):

        '''
        given the encoder output and the state, it tries to find N-best output sentences
          
        enc_out    : output from the encoder (None if an attention is not used
        enc_state  : hidden state from the encoder. 
        beam_width : beam width for search
        num_nbest  : number of final candidates
        max_len_sentence : maximum length allowed for a final sentence.

        return: Nbest of final sentences in StackDecoderPath.
        '''

        Q     = StackDecoderQueue()
        Nbest = StackDecoderQueue()

        self.num_evals = 0

        n = StackDecoderPath(0.0, [self.BOS], enc_state)
        Q.insert(n)

        while Q.size() > 0:

            n = Q.pop_highest()
            dec_input = tf.expand_dims([n.last_word()], 0)

            if self.use_attn:
                logits, dec_state_new, attn = self.decoder(dec_input, n.dec_state, enc_out)
            else:
                logits, dec_state_new = self.decoder(dec_input, n.dec_state)

            self.num_evals += 1

            # reduce dimention along batch by [0] in logits
            log_softmax = from_logit_to_log_softmax( logits[0] )

            # process the case where the sentence ends now.
            log_prob_full = n.log_prob + log_softmax.numpy()[self.EOS]

            if Nbest.size() == num_nbest and Nbest.min() < log_prob_full:
                Nbest.pop_lowest();

            if Nbest.size() < num_nbest:
                n_ready = StackDecoderPath( log_prob_full, n.sentence + [self.EOS], None )
                Nbest.insert( n_ready )

            # explore the search space with new words that would follow the current sentence.
            if len( n.sentence ) < max_len_sentence:
                for i, v in enumerate(log_softmax.numpy()):
                    if i != 0 and i != self.BOS and i != self.EOS:

                        log_prob_new = n.log_prob + v
                        sentence_new = n.sentence + [i]

                        if Q.size() == beam_width and Q.min() < log_prob_new:
                            Q.pop_lowest()

                        if Q.size() < beam_width:
                            n_new = StackDecoderPath( log_prob_new, sentence_new, dec_state_new )
                            Q.insert( n_new )

        return reversed( [ v for v in Nbest.rbtree ] )
