- The single attention mechanism although good, it is limiting. It's better for 5 people to solve a complex problem than one, even though he is talented.
- We may want our model to combine knowledge from different behaviors of the same attention mechanism.
- In Multi-head attention, instead of doing one attention calculation, we do multiple attention calculations in parallel, each with a slightly different "perspective," and then combine the results.

## Multi-Head Attention Idea
- We create h parallel attention mechanisms, called "heads."
- How do we give each head a different perspective? 
	- Before feeding the queries (q), keys (k), and values (v) into each head, we first transform them using different, learned linear projections (i.e., we multiply by different weight matrices).
	- The projections are learning different "feature subspaces".
	- For head #1, we project q, k, v using one set of learned matrices 
	  ($\large \mathbf W_1^{(q)}, \mathbf W_1^{(k)},\mathbf W_1^{(v)}$). 
	- For head #2, we use a different set ($\large \mathbf W_2^{(q)}, \mathbf W_2^{(k)},\mathbf W_2^{(v)}$), and so on.
	- This allows each head to potentially focus on different aspects of the input because it's operating on differently transformed versions of the queries, keys, and values.
- Each head i performs its own attention pooling calculation (e.g., scaled dot product attention or additive attention) using its projected queries, keys, and values. This results in h separate output vectors ($\large h_1, h_2, ..., h_h$).
- The outputs from all h heads are concatenated together into one larger vector.
- This concatenated vector is then passed through one more learned linear projection ($\large W_o$) to produce the final multi-head attention output. This final projection combines the information learned by all the different heads into a single representation.
![[MultiHeadAttention1.png]]

## Formalizing the model
- Inputs: Query q, Keys k and Values v.
- Number of heads: h.
- Calculation:
	- For each head i from 1 to h, $\huge \mathbf{h}_i = f(\mathbf W_i^{(q)}\mathbf q, \mathbf W_i^{(k)}\mathbf k,\mathbf W_i^{(v)}\mathbf v) \in \mathbb R^{p_v}$
		- Where, $\large \mathbf W_i^{(q)}, \mathbf W_i^{(k)},\mathbf W_i^{(v)}$ are learnable parameters.
		- $\large f$ is attention pooling function like additive attention.
		- $\large h_i$ is the output vector from head i.
	- Finally, we concatenate all the heads and multiply by final learnable weight matrix $\large W_o$
			- $\begin{split}\mathbf W_o \begin{bmatrix}\mathbf h_1\\\vdots\\\mathbf h_h\end{bmatrix} \in \mathbb{R}^{p_o}.\end{split}$
