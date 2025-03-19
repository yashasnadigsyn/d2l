- In general sequence-to-sequence problems like Machine Translation, inputs and outputs are of varying lengths that are unaligned.
- The standard approach to handling this sort of data is to design an encoder–decoder architecture consisting of two major components *encoder* and *decoder*.
- An encoder takes a variable-length sequence as input.
- A decoder acts as a conditional language model, taking in the encoded input and the leftwards context of the target sequence and predicting the subsequent token in the target sequence.
 ![[EncDec1.png]]