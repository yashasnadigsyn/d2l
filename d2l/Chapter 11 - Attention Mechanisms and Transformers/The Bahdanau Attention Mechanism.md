- The old seq-to-seq models had an encoder, which reads the input sequence and compresses all information into a fixed-size vector. This is also called as fixed-dimensional state. The decoder model uses this state to predict final sequence token by token.
![[BahdanauAttentionMechanism1.png]]
- This worked well for short sequences but for long sequences, it failed. 
- The core issue is the **fixed-size context vector**. It has to act as a "sufficient statistic" or summary of the entire input sequence.

## The Bahdanau Attention
- Bahdanau et al. proposed a "differentiable attention model" that allows the decoder to look beyond just the single, fixed context vector.
- The key idea is:
	- When the decoder is about to predict the next output token, it shouldn't rely only on the summary vector.
	- Instead, it should be able to look back at the entire sequence of the encoder's hidden states.
	- Also, It should dynamically decide which parts of the input sequence are most relevant for generating the current output token. This is the "attention" part – selectively focusing on relevant information.
- The key difference between the basic RNN encoder-decoder is that the context variable c is no longer fixed. Instead, a new context variable c_t' is computed at each decoding time step t'.
			$\huge \mathbf{c}_{t'} = \sum_{t=1}^{T} \alpha(\mathbf{s}_{t' - 1}, \mathbf{h}_{t}) \mathbf{h}_{t}$
- Where,
	- $\huge \mathbf{c}_{t'}$: The context vector for the current decoding step t'.
	- $\huge \mathbf{h}_{t}$: The hidden state of the encoder at input time step t (for t from 1 to T, where T is the length of the input sequence). These h_t vectors represent the input sequence at different points.
	- $\huge \mathbf{s}_{t' - 1}$: The hidden state of the decoder from the previous decoding time step t'-1.
	- $\huge \alpha(\mathbf{s}_{t' - 1}, \mathbf{h}_{t})$: The **attention weight**. This weight determines how much "attention" the decoder (in state s_(t'-1)) should pay to the t-th input word's representation (h_t) when generating the output at step t'.

- How are attention weights calculated?
	- The decoder's previous hidden state $\mathbf{s}_{t' - 1}$ acts as a query. It represents what the decoder is currently thinking about or trying to generate.
	- The encoder hidden states h_t (for all t from 1 to T) act as the **keys**. They represent the different parts of the input sequence.
	- In Bahdanau Attention Mechanism, the encoder hidden states h_t also acts as the **values**. This means the context vector c_t' will be a weighted sum of the encoder hidden states themselves. 

![[BahdanauAttentionMechanism2.png]]

