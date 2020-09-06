# TFStackDecoder
A stack decoder for Tensorflow 2.x seq2seq models that generates N-Best.

I made it to retrieve N-best predicted sentences in the descending order of probability.
The predicited sentences can be similar in terms of edit distance.
I want the same output always, given the same input, and therefore 
[tfa.seq2se2.BeamSearchDecoder] (https://www.tensorflow.org/addons/api_docs/python/tfa/seq2seq/BeamSearchDecoder)
is not suitable a it uses sampling and designed to give variety of outputs.

The output will look like this:
```
Input: <mcdonald>
Predicted translatons with probability (Nbest):
[0.1748] <マクドナルド>
[0.1044] <マクダナルド>
[0.0557] <マックドナルド>
[0.0546] <マクダノルド>
[0.0400] <マクダナード>

```

It assumes [the general RNN encoder-decoder framework](https://www.tensorflow.org/tutorials/text/nmt_with_attention).

The algorithm is based on the one given in Section 10.2 A^* ("Stack") Decoding in [JM09]
(https://www.pearson.com/us/higher-education/program/Jurafsky-Speech-and-Language-Processing-2nd-Edition/PGM181706.html).
# Dependencies

* [Tensorflow 2.x](https://www.tensorflow.org/)

* [RedBlackTree](https://github.com/ShoYamanishi/RedBlackTree)


# Contact
For any inquiries, please contact:
Shoichiro Yamanishi

yamanishi72@gmail.com

# Usage:

```python
from TFStackDecoder.stack_decoder import StackDecoder
from TFStackDecoder.stack_decoder import StackDecoderPath

# This is a replacement to evaluate() defined in the NMT tutorial
# It does not generate for attention plots.
# It generates N-best sentences in the descending order of probability.

def evaluate_nbest(sentence):

  sentence = preprocess_sentence(sentence)

  inputs = [inp_lang.word_index[i] for i in sentence.split(' ')]
  inputs = tf.keras.preprocessing.sequence.pad_sequences([inputs],
                                                         maxlen=max_length_inp,
                                                         padding='post')
  inputs = tf.convert_to_tensor(inputs)

  hidden = [tf.zeros((1, units))]
  enc_out, enc_hidden = encoder(inputs, hidden)

  dec_hidden = enc_hidden
  dec_input = tf.expand_dims([targ_lang.word_index['<start>']], 0)

  BOS = [targ_lang.word_index['<start>']
  EOS = [targ_lang.word_index['<end>']

  # Here decoder is assumed to have been defined as a GRU decoder
  # with BahdanauAttention.
  stack_decoder = StackDecoder(decoder, BOS, EOS, use_attn=True)

  BEAM_WIDTH = 20
  NUM_NBEST = 5
  MAX_LEN = max_length_targ + 2 # this is defined earlyl in the NMT tutorial.
                                # + 2 is for BOS and EOS.
  nbest_list = stack_decoder.NBest( enc_out, enc_hidden, BEAM_WIDTH, NUM_NBEST, MAX_LEN )

  results = []
  for r in nbest_list:
    result = []
    for i in r.sentence:
      result.append( targ_lang.index_word[i] )
    result = ' '.join(result)
    results.append((r.log_prob, result))

  return results, sentence
```

# References

* NMT : [Neural machine translation with attention](https://www.tensorflow.org/tutorials/text/nmt_with_attention)

* JM09 : Speech and Language Processing, 2nd., Daniel Jurafsky & James H. Martin, 2009 Pearson Education