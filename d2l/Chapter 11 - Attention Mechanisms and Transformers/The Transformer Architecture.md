## Model Architecture
![[TransformerArchitecture1.png]]

- The Transformer follows the standard **encoder-decoder architecture**, but the internal structure of the encoder and decoder is different.
### Encoder
- Composed of multiple identical layers stacked on top of each other.
- Each Encoder layer has two sublayers:
	- **Multi-Head Self-Attention:** This is the core. Queries, Keys, and Values all come from the output of the previous encoder layer. This allows each position in the sequence to attend to all other positions in the sequence (at that layer's level of representation).
	- **Positionwise Feed-Forward Network (FFN):** A simple fully connected feed-forward network (usually two linear layers with a ReLU or similar activation in between). Importantly, this FFN is applied **independently to each position**. The same FFN weights are used for all positions, but it processes each position's vector separately.
- **Residual Connections & Layer Normalization:**
	- Around each of the two sublayers, there's a **residual connection**.
	- $output = x + sublayer(x)$. 
	- This is like the skip connections in ResNet, helping with gradient flow and enabling deeper networks.
	- After the addition, **Layer Normalization** is applied. Layer Normalization stabilizes the activations within each layer.
- **Output:** The encoder outputs a sequence of vectors (one for each input position) representing the input sequence enriched with contextual information.

### Decoder
- Similar to the encoder, it's a stack of identical layers.
- **Each Decoder Layer has Three Sublayers:**
	- **Masked Multi-Head Self-Attention:** Similar to the encoder's self-attention, but with a crucial difference: it's **masked**. Masking ensures that when predicting the output token at position i, the decoder can only attend to positions up to and including i in the decoder's input. It cannot "look ahead" at future tokens in the output sequence. This preserves the **autoregressive property** needed for generation (predicting one token at a time based on previous ones). Queries, Keys, and Values come from the output of the previous decoder layer.
	- **Multi-Head Encoder-Decoder Attention:** This is where the decoder interacts with the encoder's output.
        - **Queries:** Come from the output of the previous decoder sublayer (the masked self-attention layer).
        - **Keys and Values:** Come from the **output of the encoder stack**. This allows each position in the decoder to attend to all positions in the input sequence (as represented by the encoder's output).
	- **Positionwise Feed-Forward Network (FFN):** Identical in structure and function to the FFN in the encoder, applied position-wise to the output of the encoder-decoder attention sublayer.
- **Residual Connections & Layer Normalization:** Applied around each of the three sublayers, just like in the encoder.

## The Rest is completely code which can be found in:
https://yashasnadigsyn.github.io/d2l