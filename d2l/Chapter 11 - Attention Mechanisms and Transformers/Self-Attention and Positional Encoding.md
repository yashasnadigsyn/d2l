## Self-Attention
- Till now, we have been only using encoder-decoder models, where in RNNs decoder uses context variable c which is a fixed dimensional state, whereas, in case of Bahdanau Transformers, we have seen decoder models attending each hidden state in encoder.
- But, What if we apply the attention mechanism within a single sequence?
	- The idea is to allow each token in a sequence to attend to all other tokens (including itself) in the same sequence.
- To compute a new representation for each token that incorporates information and context from the entire sequence, weighted by relevance.
- We have an input sequence $\large x_1, ..., x_n$, where each $\large x_i$ is a vector (e.g., a word embedding).
- Self-attention produces an output sequence $\large y_1, ..., y_n$ of the same length, where each $\large y_i$ is a new representation for the corresponding input token $\large x_i$.

				$\huge \mathbf{y}_i = f(\mathbf{x}_i, (\mathbf{x}_1, \mathbf{x}_1), \ldots, (\mathbf{x}_n, \mathbf{x}_n)) \in \mathbb{R}^d$

- Calculation:
	- Remember, Attention formula:
		- $\huge Attention(q, D) = ∑ α(q, k_j) * v_j)$
		- **Query ($q$):** For calculating the output y_i, the query comes from the input token x_i itself (or a projection of it).
		- **Keys ($k_j$):** The keys come from all input tokens x_j (for j from 1 to n).
		- **Values ($v_j$):** The values also come from all input tokens x_j (for j from 1 to n).
	- To calculate the new representation $\large y_i$ for token $\large x_i$,
		- $\large x_i$ generates a query.
		- Every token $\large x_j$ (including $\large x_i$ itself) generates a key and a value.
		- The query from $\large x_i$ is compared to the keys from all $\large x_j$ to compute attention weights $\large α(q_i, k_j)$.
		- $\large y_i$ is calculated as the weighted sum of all values $\large v_j$, using the computed attention weights: $\large y_i = ∑_j α(q_i, k_j) * v_j$.
## Comparison of CNNs, RNNs and Self-Attention

![[SelfAttention1.png]]

- Let's compare the these three architectures for sequential encoding (mapping an input sequence to an output sequence of the same length).
- We look at the below three comparisons:
	- Computational Complexity: How much computation is needed?
	- Sequential Operations: How many steps depend on the previous step's output (limits parallelism)?
	- Maximum Path Length: What's the maximum number of steps information has to travel between any two positions in the sequence (affects learning long-range dependencies)?

- Convolutional Neural Networks (CNNs - 1D for sequences):
	- Analogy: Treats a sequence like a 1D image, using kernels to process local features (like n-grams).
	- Complexity: $\large O(k * n * d^2)$ - Linear in sequence length n, depends on kernel size k and feature dimension d.
	- Sequential Operations: $\large O(1)$ - Convolutions over different parts of the sequence can be done in parallel.
	- Max Path Length: $\large O(n / k)$ - Information needs to pass through multiple layers to connect distant positions. The path length grows logarithmically with n if using dilated convolutions, but linearly with standard convolutions stacked. The path length depends on the kernel size k and the number of layers.

- Recurrent Neural Networks (RNNs):
	- Mechanism: Processes the sequence step-by-step, maintaining a hidden state.
	- Complexity: $\large O(n * d^2)$ - Linear in sequence length n, depends quadratically on hidden dimension d.
	- Sequential Operations: $\large O(n)$ - Each step depends on the hidden state from the previous step. This calculation is inherently sequential and cannot be fully parallelized over the time dimension.
	- Max Path Length: $\large O(n)$ - Information from the beginning of the sequence has to travel through n steps to reach the end. This makes learning long-range dependencies difficult (vanishing/exploding gradients).

- Self-Attention:
	- Mechanism: Directly computes interactions between all pairs of positions using attention.
	- Complexity: $\large O(n^2 * d)$ - Quadratic in sequence length n, linear in feature dimension d. This is the main drawback – it becomes very expensive for long sequences. The dominant cost comes from computing the n x n attention score matrix ($\large QK^T$).
	- Sequential Operations: $\large O(1)$ - The calculation for each position can be done in parallel once the initial queries, keys, and values are computed. There are no step-by-step dependencies like in RNNs.
	- Max Path Length: $\large O(1)$ - Each token is directly connected to every other token via the attention mechanism in a single step. Information can flow between any two positions in just one step, making it excellent for capturing long-range dependencies.

- Comparison Summary (Figure 11.6.1):
	- Parallelism: CNNs and Self-Attention allow for much more parallel computation than RNNs.
	- Long-Range Dependencies: Self-Attention has the shortest maximum path length ($\large O(1)$), making it theoretically best suited for learning long-range dependencies. RNNs have the longest ($\large O(n)$), making it hardest. CNNs are in between.
	- Computational Cost vs. Length: Self-Attention's $\large O(n^2 * d)$ complexity is its Achilles' heel for very long sequences, compared to the linear $\large O(n * ...)$ complexity of RNNs and CNNs.

## Positional Encoding
- RNNs handle order naturally. RNNs process sequences token by token, recurrently. This sequential processing inherently incorporates the order information into the hidden state. The hidden state at time t depends on the state at t-1 and the input at t.
- Self-Attention Loses Order. Self-attention, on the other hand, computes attention scores between all pairs of tokens simultaneously (or in parallel). 
- If we just feed word embeddings into a self-attention layer, it treats the input like a "bag of words" – it knows which words are present but not where they are in the sequence. Permuting the input sequence would result in exactly the same output representations for each token (just permuted), which is clearly wrong for language where word order is critical.
- To solve this, we need to explicitly provide information about the position of each token in the sequence to the model. This is done using positional encodings.
- The standard approach is to create a positional encoding vector for each position in the sequence. This vector has the same dimension (d) as the token embeddings. The positional encoding vector for position i is then added to the token embedding for the word at position i.
- **Fixed vs. Learned:** Positional encodings can be either:
    - **Learned:** Treated as parameters of the model and learned during training (similar to word embeddings).
    - **Fixed:** Calculated using a predefined formula, not learned. 

			$\huge \begin{split}\begin{aligned} p_{i, 2j} &= \sin\left(\frac{i}{10000^{2j/d}}\right),\\p_{i, 2j+1} &= \cos\left(\frac{i}{10000^{2j/d}}\right)\end{aligned}\end{split}$
			
- Basically, for each position (i,j) in the embedding matrix, we create another matrix P of the same dimension where each position has a unique number which represents the position of the token.
- Using the above formulas, we can create absolute and relational positional encodings.
	- Where, absolute positional encoding provides the absolute position of the token.
	- And, Relational positional encoding provides relation between two tokens in a sequence.