- This section is about decoding strategies for sequence to sequence models, especially for generating the output sequence during inference.
- There are three different approaches:
	- **Greedy Search**
	- **Exhaustive Search**
	- **Beam Search**

#### Greedy Search
- In previous sections, we built a seq-to-seq model which at each time t' predicts a probabilities of all possible tokens in the vocabulary. 
- Greedy search simply picks the token y_t' with the **highest conditional probability** P(y | y_1, ..., y_(t'-1), c).
- The generation stops when the decoder produces the \<eos> token or reaches a predefined maximum sequence length T'.
- Consider the below two figures:
  ![[BeamSearch1.png]]
  ![[BeamSearch2.png]]
- The conditional probability of the 1st one is 0.048 and the second one is 0.054.
- In the first one, we followed greedy method and chose the highest probability at each time. This may seem better but when we look for some alternative path (like the second one), we can get better sequence than first one.

#### Exhaustive Search
- Exhaustive search aims to find the output sequence that has the **highest joint probability.** This would be the truly optimal sequence if the model's probabilities were perfectly accurate.
- Exhaustive search considers every single possible output sequence of length up to T' and calculates its probability. Then, it chooses the sequence with the highest probability.
- This is computationally expensive.
  
#### Beam Search
- Beam search strikes a balance between the efficiency of greedy search and the optimality of exhaustive search.
- Beam search is controlled by a hyperparameter called the **beam size, k**.
- Instead of just keeping track of the single best token at each step (like greedy search), beam search maintains a "beam" of k **candidate output sequences** at each time step.
  ![[BeamSearch3.png]]
- Let's understand the above example with k=2 and Max Length=3
	- **Step 1:** Top 2 tokens are A and C. Candidates: [A], [C].
	- **Step 2:** Extend [A] and [C] with all vocabulary tokens. Calculate probabilities like P(A, y_2 | c) and P(C, y_2 | c). Suppose the top 2 sequences among all extensions are [A, B] and [C, E]. New candidates: [A, B], [C, E].
    - **Step 3:** Extend [A, B] and [C, E]. Calculate probabilities like P(A, B, y_3 | c) and P(C, E, y_3 | c). Suppose top 2 are [A, B, D] and [C, E, D]. New candidates: [A, B, D], [C, E, D].
    - **Final Candidates:** After 3 steps (or when \<eos> is generated in some paths), we have a set of candidate sequences (A, C, A-B, C-E, A-B-D, C-E-D).
- In the end, we obtain the set of final candidate output sequences based on these six sequences (e.g., discard portions including and after “\<eos>”). Then we choose the output sequence which maximizes the following score:
	- $\frac{1}{L^\alpha} \log P(y_1, \ldots, y_{L}\mid \mathbf{c}) = \frac{1}{L^\alpha} \sum_{t'=1}^L \log P(y_{t'} \mid y_1, \ldots, y_{t'-1}, \mathbf{c});$
- Here 𝐿 is the length of the final candidate sequence and 𝛼 is usually set to 0.75. Since a longer sequence has more logarithmic terms in the summation, the term 𝐿𝛼 in the denominator penalizes long sequences.