- Sometimes context from both direction is important than just predicting next word.
- Example:
	- I am _ _ _ _ _ _. 
	- I am _ _ _ _ _ _ hungry.
	- I am _ _ _ _ _ _ hungry, and I can eat half a pig.

- In the first sentence, "happy" seems to be likely candidate.
- In the second one, both "very" and "not" fits the context.
- In the third one, "very" feels much better.

- This is where Bidirectional RNNs comes in. Tasks like POS tagging and Masked Language Modeling (the above examples) need much more than unidirectional RNNs.
- The core idea is to use two RNNs from each side.
![[Bidirection1.png]]
- For the first RNN layer, the first input is x1 and the last input is xT , but for the second RNN layer, the first input is xT and the last input is x1. 
- To produce the output of this bidirectional RNN layer, we simply concatenate together the corresponding outputs of the two underlying unidirectional RNN layers.
- Formally,
	- $\overrightarrow{\mathbf{H}}_t = \phi(\mathbf{X}_t \mathbf{W}_{\textrm{xh}}^{(f)} + \overrightarrow{\mathbf{H}}_{t-1} \mathbf{W}_{\textrm{hh}}^{(f)}  + \mathbf{b}_\textrm{h}^{(f)})$
	- $\overleftarrow{\mathbf{H}}_t = \phi(\mathbf{X}_t \mathbf{W}_{\textrm{xh}}^{(b)} + \overleftarrow{\mathbf{H}}_{t+1} \mathbf{W}_{\textrm{hh}}^{(b)}  + \mathbf{b}_\textrm{h}^{(b)})$
- and, Output is,
	- $\mathbf{O}_t = \mathbf{H}_t \mathbf{W}_{\textrm{hq}} + \mathbf{b}_\textrm{q}$
