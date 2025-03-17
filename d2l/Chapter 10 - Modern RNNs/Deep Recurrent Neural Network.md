- RNNs are deep in the way they are deep in time which means hidden state at time step t can also affect hidden state at time step n. This is called "Depth in Time".
- But, MLPs or CNNs are deep in the way where they have many hidden layers stacked upon each other. This is called "Depth in Layers".
- Deep RNNs combine both to form "Depth in Time and Layers".
![[DRNN1.png]]
- **Input Sequence:** It is a input sequence X at the bottom.
- **Layer 1 RNN:** The first RNN layer processes this input sequence. It takes the input at each time step and its own previous hidden state (recurrent connection) and produces an output sequence.
- **Layer 2 RNN:** The output sequence from Layer 1 becomes the input sequence for Layer 2. Layer 2 processes this new sequence in the same recurrent manner, producing its own output sequence.
- **Layer 3, Layer 4, ... Layer L RNN:** This stacking continues for L layers. Each layer takes the output sequence from the layer below as its input sequence.
- **Output Layer:** Finally, the output sequence from the topmost RNN layer (Layer L) is used to calculate the final output of the entire deep RNN.

- Main Difference is **Hierarchical Feature Extraction**. Just like in deep CNNs, stacking RNN layers allows the network to learn features in a hierarchical manner.
	- **Lower Layers (closer to input):** Might learn more basic, lower-level features of the sequence. For example, in text, the first layer might learn about individual words or characters and their basic properties.
    - **Higher Layers (further from input):** Can learn more abstract, higher-level features based on the features extracted by the lower layers. In text, higher layers might learn about phrases, sentences, paragraphs, and even the overall context or meaning.

- Formally, Each RNN cell depends on both the same layer’s value at the previous time step and the previous layer’s value at the same time step.
	- $\mathbf{H}_t^{(l)} = \phi_l(\mathbf{H}_t^{(l-1)} \mathbf{W}_{\textrm{xh}}^{(l)} + \mathbf{H}_{t-1}^{(l)} \mathbf{W}_{\textrm{hh}}^{(l)}  + \mathbf{b}_\textrm{h}^{(l)})$
	  
- At the output layer,
	- $\mathbf{O}_t = \mathbf{H}_t^{(L)} \mathbf{W}_{\textrm{hq}} + \mathbf{b}_\textrm{q}$