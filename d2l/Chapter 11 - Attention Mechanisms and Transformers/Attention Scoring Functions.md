- This section dives into different ways to calculate the "compatibility" or "relevance" between queries and keys, which determines the attention weights.
- Normally, we use distance measure to find similarity. Example: Gaussian, where points nearer are given more weights than far points and weights decrease gradually.
- But, Dot Products are computationally cheaper than distance measures.
- Goal: As discussed in Attention ([[Queries,  Keys and Values]]), Our goal is to find attention scoring functions $\alpha(\mathbf{q}, \mathbf{k}_i)$ that are:
	- Computationally Efficient
	- Good Relevancy

![[AttentionScoringFunctions1.png]]

## Dot Product Attention
- Let’s review the attention function (without exponentiation) from the Gaussian kernel for
a moment:
		$\huge a(\mathbf{q}, \mathbf{k}_i) = -\frac{1}{2} \|\mathbf{q} - \mathbf{k}_i\|^2  = \mathbf{q}^\top \mathbf{k}_i -\frac{1}{2} \|\mathbf{k}_i\|^2  -\frac{1}{2} \|\mathbf{q}\|^2$

- **Simplification 1** (Dropping Constant Term - $-\frac{1}{2} \|\mathbf{q}\|^2$)
	- This term depends only on query q, which is same for all (key, value) pairs for a given query.
	- When we apply the softmax to normalize the attention weights, adding or subtracting a constant value from all scores $\alpha(\mathbf{q}, \mathbf{k}_i)$ before the softmax doesn't change the relative probabilities. The softmax is scale-invariant to shifts in the input.
	- Conclusion: We can drop the term $-\frac{1}{2} \|\mathbf{q}\|^2$ without changing the final attention weights after softmax normalization.

- **Simplification 2** (Dropping the term - $-\frac{1}{2} \|\mathbf{k}_i\|^2$)
	- Normalization has become a norm and we use many different types of normalization like layer or batch normalization.
	- Imagine we have a set of keys like k1, k2, k3 (which are vectors) etc.. 
		- Without Normalization, the length of these vectors will vary wildly. One can have 10 values and another can have 10000 values in the vector.
		- With normalization, the length of the vectors are all similar.
	- The main reason we use normalization like Batch Normalization or Layer Normalization is that, they make training more stable and less prone to issues like vanishing or exploding gradients.
	- Another reason is, they tend to keep the norms of the vectors (like k_i in our case) within a reasonable range and prevent them from becoming arbitrarily large or small.
	- If the norms || k_i || of all keys k_i are approximately constant (which means, the length of all the key vectors k_i are similar), then, the term $-\frac{1}{2} \|\mathbf{k}_i\|^2$ becomes approximately constant as well.
	- And, if $-\frac{1}{2} \|\mathbf{k}_i\|^2$ is approximately constant for all i, then just like with the $-\frac{1}{2} \|\mathbf{q}\|^2$ term, it becomes less important for determining the relative attention weights after softmax. The softmax is primarily influenced by the differences in scores, not by a constant offset that's the same for all scores.

- After dropping the above two terms, we are left with the dot product $\mathbf{q}^\top \mathbf{k}_i$ as the core component of the scoring function. Dot product is a measure of similarity, especially if vectors are normalized.
- If elements of q and k_i are independent and identically distributed with mean 0 and variance 1, the dot product $\mathbf{q}^\top \mathbf{k}_i$ has a variance of d. As the dimension d increases, the variance of the dot product also increases.
- To keep the variance of the dot product roughly constant (around 1), regardless of the dimension d, the dot product is scaled down by $1/\sqrt{d}$. This scaling helps to prevent the dot products from becoming too large or too small, especially in high-dimensional spaces, which can affect the softmax output.
- Now, the attention scoring function has become:
			$\huge a(\mathbf{q}, \mathbf{k}_i) = \mathbf{q}^\top \mathbf{k}_i / \sqrt{d}$

- Finally, 
			$\huge \alpha(\mathbf{q}, \mathbf{k}_i) = \mathrm{softmax}(a(\mathbf{q}, \mathbf{k}_i)) = \frac{\exp(\mathbf{q}^\top \mathbf{k}_i / \sqrt{d})}{\sum_{j=1} \exp(\mathbf{q}^\top \mathbf{k}_j / \sqrt{d})}$
- The scaled dot product scores are then passed through the softmax function to get the final attention weights. This ensures that the weights are non-negative and sum to 1, forming a probability distribution over the keys.
- 