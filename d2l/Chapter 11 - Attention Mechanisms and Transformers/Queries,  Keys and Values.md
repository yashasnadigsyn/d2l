- While deep learning in 2010s was largely driven by MLPs, CNNs, and RNNs, the architecture itself hadn't changed drastically from older neural network concepts.
- Innovations were more in training techniques (ReLU, BatchNorm, etc.) and leveraging more compute and data.
- **Transformer** architecture was a big fundamental shift.
- The "core idea" behind Transformers is the **attention mechanism**. This was initially proposed as an enhancement for encoder-decoder RNNs in machine translation.
- The limitation of basic encoder-decoder RNNs where the entire input sequence is compressed into a single, fixed-length context vector. The problem: for long sequences, compressing everything into one vector can lose information and become a bottleneck.
- The key intuition behind attention is to allow the **decoder to "revisit" the input sequence** at every decoding step. Instead of relying solely on a single, fixed context vector, the decoder should be able to dynamically focus on relevant parts of the input sequence as it generates each output token.
- The attention mechanism is inspired by database on some key points:
	- **Queries work regardless of database size:** We can use the same query, whether the database has 6 entries or millions.
	- **Different answers based on database content:** If "Key" is in the database, you get "Value"; if not, you get no valid answer.
	- **Simple "code" for complex operations:** Operations on a large database can be simple (exact match, approximate match, top-k retrieval).
	- **No need to compress the database:** We don't need to simplify or compress the database to make queries effective.

- Attention is like "querying" a database of (key, value) pairs. 
	- Database D: $\mathcal{D} \stackrel{\textrm{def}}{=} \{(\mathbf{k}_1, \mathbf{v}_1), \ldots (\mathbf{k}_m, \mathbf{v}_m)\}$, A set of (key, value) pairs.
	- Query q: Something we want to look up in the database D.

- Attention operation
	- $\huge \textrm{Attention}(\mathbf{q}, \mathcal{D}) \stackrel{\textrm{def}}{=} \sum_{i=1}^m \alpha(\mathbf{q}, \mathbf{k}_i) \mathbf{v}_i$
	- Where,
		- $\textrm{Attention}(\mathbf{q}, \mathcal{D})$: The output of the attention mechanism.
		- $\alpha(\mathbf{q}, \mathbf{k}_i)$: Attention weight for the i-th (key, value) pair. It's a scalar value that represents how much "attention" to pay to the i-th value v<sub>i</sub> given the query q and the i-th key k<sub>i</sub>.
		- $\mathbf{v}_i$: The i-th value from the database.
		- $\sum_{i=1}^m$: Summation over all (key, value) pairs in the database.

- In simple words, Attention is a **weighted sum** of the **values** in the database. The **weights** (α(q, k_i)) are determined by how "compatible" the **query** q is with each **key** k_i. If a key k_i is very relevant to the query q, its corresponding weight α(q, k_i) will be high, and its value v_i will contribute more to the attention output.
- The operation is called **attention pooling** because it pools (combines) the values based on attention weights.

- Special Cases of Attention Weights:
	- **Non-negative Weights (α(q, k_i) ≥ 0):** Output is in the "convex cone" of values.
	- **Convex Combination Weights (∑_i α(q, k_i) = 1 and α(q, k_i) ≥ 0):** Most common in deep learning. Weights sum to 1 and are non-negative. Output is a weighted average of values.
	- **Exact Match (One weight = 1, others = 0):** Like a traditional database query – retrieves a specific value associated with a matching key.
	- **Average Pooling (All weights = 1/m):** Weights are equal. Attention output is just the average of all values.

- A common strategy for ensuring that the weights sum up to 1 is to normalize them via

				$\huge \alpha(\mathbf{q}, \mathbf{k}_i) = \frac{\alpha(\mathbf{q}, \mathbf{k}_i)}{{\sum_j} \alpha(\mathbf{q}, \mathbf{k}_j)}$

- In particular, to ensure that the weights are also non-negative, one can resort to exponentiation. This means that we can now pick _any_ function $a(\mathbf{q}, \mathbf{k})$ and then apply the softmax operation used for multinomial models to it via

				$\huge \alpha(\mathbf{q}, \mathbf{k}_i) = \frac{\exp(a(\mathbf{q}, \mathbf{k}_i))}{\sum_j \exp(a(\mathbf{q}, \mathbf{k}_j))}$

- Softmax is differentiable and its gradient doesn't vanish easily, which are desirable properties for training neural networks with backpropagation.

	![[Queries1.png]]


