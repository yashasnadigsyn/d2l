- Again, We start this section by saying, Machine Translation is a **sequence-to-sequence** problem where both input (source sentence) and output (target sentence) are variable-length sequences.
- We use Encoder-Decoder ([[Encoder-Decoder Architecture]]) Architecture for these types of problems.
	- **Encoder:** Encodes the input sequence into a meaningful representation.
	- **Decoder:** Decodes this representation to generate the output sequence.
- We use RNNs to implement both encoder and decoder.
- RNN Encoder
	- Take a variable-length input sentence.
	- Transform it into a fixed-shape hidden state. This hidden state is often called the **context vector** or **thought vector**. This is the compressed representation of the entire input sentence, capturing its essential meaning.
- RNN Decoder:
	- Generate the output sentence, one token at a time.
	- Conditioned on:
		- The encoded input
		- The tokens it has already generated in the target sentence.
- Training vs. Testing (for decoder):
	- Training:
		- During training, when the decoder is predicting the t-th word in the target sentence, it is given the correct (t-1)-th word from the actual target sentence (the "ground truth" label) as input. This is called "teacher forcing".
		- It makes training more stable and faster.
	- Testing:
		- At test time, we don't have the "ground truth" target sentence. So, when the decoder wants to predict the t-th word, it has to use the word it itself predicted as the (t-1)-th word as input. This is called "free generation" or "autoregressive generation."
		- The decoder's output becomes its own input for the next step, creating a chain of predictions to generate the entire target sentence.

- **Teacher Forcing**
	- This is a kind of language modeling approach.
	- In Teacher Forcing, We take the original target sequence and prepend the \<bos> (beginning-of-sequence) token to it. Then, we remove the last token from this prepended sequence. This becomes the input to the decoder.
	- The "labels" (what the decoder is supposed to predict) are simply the original target sequence, but shifted by one position to the left. In effect, it's the original target sequence excluding the \<bos> token but including the \<eos> (end-of-sequence) token.
	- Example:
	  - **Target Sentence:** "Ils regardent ." (French)
	  - **Decoder Input (Teacher Forcing):** \<bos>, "Ils", "regardent", "."
	  - **Decoder Output (Labels):** "Ils", "regardent", ".", \<eos>

- **Encoder**
	- The function of encoder is to transform variable length input sequence to fixed-shape context variable 'c'.
	- Consider a single input sequence example (batch size 1): x1, x2, ..., xT, where xt is the t-th token.
	- Then, $\mathbf{h}_t = f(\mathbf{x}_t, \mathbf{h}_{t-1})$.
	- In general, $\mathbf{c} =  q(\mathbf{h}_1, \ldots, \mathbf{h}_T)$.
	- The context variable c is just the hidden state hT corresponding to the encoder RNN’s representation after processing the final token of the input sequence.

- **Decoder**
	- The decoder's main job is to produce the target output sequence token by token.
	- The decoder aims to predict the probability of each possible next token in the target sequence, conditioned on:
		- Previous tokens in the target sequence (y1, ..., y_t').
		- The context variable (c).
	- This is expressed as: $P(y_{t'+1} \mid y_1, \ldots, y_{t'}, \mathbf{c})$
	- To calculate the decoder's hidden state s_(t') at the current time step t', the RNN takes the previous target token's feature vector y_(t'-1), the context vector c (representing the source sentence), and the decoder's previous hidden state s_(t'-1). It combines these using the function g to update the hidden state.
	- $\mathbf{s}_{t^\prime} = g(y_{t^\prime-1}, \mathbf{c}, \mathbf{s}_{t^\prime-1}).$
	- Then, to predict the next token, we can use an output layer and the softmax operation to compute the predictive distribution 𝑝(𝑦𝑡′+1 | 𝑦1, . . . , 𝑦𝑡′ , c) over the subsequent output token 𝑡′ + 1.

![[Seq-to-Seq1.png]]

- **Evaluation of a predicted sequence:**
	- How can we measure how "good" a predicted sequence is compared to the correct "target" sequence? 
	- It's not as simple as checking for exact word-for-word matches because translations can be phrased differently and still be correct.
	- BLEU is introduced as a widely used metric, originally designed for machine translation, but applicable to evaluating the quality of output sequences in various tasks.
	- BLEU's main principle is to check for **n-gram matches** between the predicted sequence and the target sequence.
	- Look at this for further explanation: [[https://d2l.ai/chapter_recurrent-modern/seq2seq.html#evaluation-of-predicted-sequences]]
	  