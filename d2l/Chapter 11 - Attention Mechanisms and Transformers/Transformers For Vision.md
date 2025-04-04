- For a long time, Convolutional Neural Networks (CNNs) were the undisputed champions for computer vision tasks. They are excellent at capturing local spatial features and building hierarchical representations.
- Meanwhile, Transformers became the dominant architecture in Natural Language Processing, proving incredibly effective at capturing long-range dependencies in sequential data.
- Researchers started asking: Could the Transformer architecture, particularly its powerful self-attention mechanism, be adapted to image data and potentially outperform CNNs?
- The ViT paper (Dosovitskiy et al., 2021) presented a more direct and scalable way to apply the standard Transformer architecture to images.
- ViTs demonstrated excellent scalability. When trained on large datasets with large models, they surpassed state-of-the-art CNNs (like ResNets) significantly, indicating that Transformers could be a new foundation for computer vision, just like they became for NLP.

## Model Architecture
- The model architecture is divided into three parts: stem, body and head.

![[TransformersForVision1.png]]

### Stem
- Transformers expect a sequence of input embeddings (like word embeddings in NLP). How do you turn a 2D image into a sequence?
- We do image patchification. Divide the input image (height h, width w) into a grid of smaller, non-overlapping patches. Each patch has size p x p (e.g., 16x16 pixels). 
- Each p x p patch (which has c color channels) is flattened into a single long vector of dimension c * p * p.
- The image is transformed into a sequence of m = (h\*w) / (p\*p) patch vectors. Each patch is now treated like a "token" in a sequence.
- Each flattened patch vector is linearly projected into the Transformer's working dimension (d, often called num_hiddens). This is done using a learnable weight matrix, similar to how word indices are mapped to embeddings in NLP. This step creates the "patch embeddings."
- We add a learnable \<cls> token at the start. Now, the sequence length becomes m+1.
- The final output representation corresponding to this specific \<cls> token after it passes through the Transformer encoder will be used for the final classification decision. It acts as an aggregate representation of the entire image, informed by its attention to all patches.
- We, then, add positional embeddings like in Transformers for NLP.
- We use learnable positional embedding, not, the fixed sine/cosine one, we used in the encoder-decoder Transformer.

### Body
- The resulting sequence of m + 1 vectors is fed into a standard **Transformer Encoder** stack (multiple layers).
- Each layer consists of Multi-Head Self-Attention followed by a Positionwise Feed-Forward Network, with residual connections and layer normalization around each sub-layer.
- Within each multi-head self-attention layer, every element in the sequence (each patch embedding and the \<cls> token) attends to every other element. This allows the model to capture **global relationships** between different parts of the image, going beyond the local receptive fields typical of early CNN layers.
- The encoder outputs m + 1 vectors of dimension d, representing the final contextualized embeddings for each patch and the \<cls> token.

### Head
- Only the output vector corresponding to the \<cls> token from the final encoder layer is taken. This vector is assumed to have aggregated the necessary global image information for classification via the self-attention mechanism.
- This \<cls> token representation is passed through a final classification head (usually a simple MLP, often just a single linear layer) to produce the final class probabilities or logits.