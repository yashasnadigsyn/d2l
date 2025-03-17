- RNNs and especially LSTMs became very successful in the 2010s for handling sequential data (like text, time series, etc.). They were great at remembering information over time. However, LSTMs, while powerful, can be computationally expensive.
- The GRU architecture was developed as a more computationally efficient alternative to LSTMs. The goal was to simplify the LSTM's gating mechanism while retaining its ability to capture long-range dependencies and achieve comparable performance in many sequence modeling tasks.
## Reset Gate and Update Gate
- Here, the LSTM’s three gates are replaced by two: the reset gate and the update gate. 
- The reset gate controls how much of the previous state we still want to remember.
- The update gate controls how much the new input and old hidden state to be in new hidden state.
![[GRU1.png]]
- Mathematically, 
	- $\mathbf{R}_t = \sigma(\mathbf{X}_t \mathbf{W}_{\textrm{xr}} + \mathbf{H}_{t-1} \mathbf{W}_{\textrm{hr}} + \mathbf{b}_\textrm{r})$
	- $\mathbf{Z}_t = \sigma(\mathbf{X}_t \mathbf{W}_{\textrm{xz}} + \mathbf{H}_{t-1} \mathbf{W}_{\textrm{hz}} + \mathbf{b}_\textrm{z})$

## Candidate Hidden State
- The "candidate hidden state" is a "proposed" new hidden state based on the current input and some information from the past (Ht-1). It's not the final hidden state yet, but just a proposal.
 ![[GRU2.png]]
 - $\tilde{\mathbf{H}}_t = \tanh(\mathbf{X}_t \mathbf{W}_{\textrm{xh}} + \left(\mathbf{R}_t \odot \mathbf{H}_{t-1}\right) \mathbf{W}_{\textrm{hh}} + \mathbf{b}_\textrm{h})$
 
 - **Effect of Reset Gate:**
	 - If Rt is close to 1: (R_t ⊙ H_t-1) is almost the same as H_t-1. We are using a lot of the previous hidden state to calculate the candidate. We are "remembering" the past.
	- If R_t is close to 0: (R_t ⊙ H_t-1) becomes close to zero. We are effectively "resetting" or ignoring the previous hidden state. The candidate state is then mainly based on the current input X_t.

## Hidden State
- The candidate hidden state H_t is just a candidate. The update gate Z_t determines how much of this candidate state to actually use and how much of the previous hidden state H_t-1 to keep. This is the final step in calculating the new hidden state H_t.
 ![[GRU3.png]]
 - $\mathbf{H}_t = \mathbf{Z}_t \odot \mathbf{H}_{t-1}  + (1 - \mathbf{Z}_t) \odot \tilde{\mathbf{H}}_t.$
 - Whenever the update gate Zt is close to 1, we simply retain the old state. In this case the information from Xt is ignored, effectively skipping time step t in the dependency chain. 
 - By contrast, whenever Zt is close to 0, the new latent state Ht approaches the candidate latent state $\tilde{\mathbf{H}}_t$.

## Conclusion
- In summary, GRUs have the following two distinguishing features:
	- Reset gates help capture short-term dependencies in sequences.
	- Update gates help capture long-term dependencies in sequences.